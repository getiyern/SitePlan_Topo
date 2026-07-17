"""
app.py -- Streamlit UI for the terrain-informed, blank-slate site-plan engine.

Enter any Kendall County, Illinois address (or click the map) and the app finds
the underlying parcel, pulls its real USGS 3DEP terrain, and develops the parcel
from a blank slate: contour-aligned roads and buildings, a terrain-seeking
stormwater pond, and real cut/fill earthwork economics.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import os

import folium
import matplotlib
matplotlib.use("Agg")
import streamlit as st
from folium.plugins import Draw
from shapely.geometry import mapping
from streamlit_folium import st_folium

import siteplan_engine as E

HERE = os.path.dirname(os.path.abspath(__file__))
PARQUET = os.path.join(HERE, "parcels.parquet")

# Kendall County, IL approximate center + bounds for the locator map.
COUNTY_CENTER = (41.5900, -88.4300)
COUNTY_BOUNDS = [[41.35, -88.68], [41.75, -88.19]]

st.set_page_config(page_title="Kendall County Site Planner",
                   page_icon="🏗️", layout="wide")


# ---------------------------------------------------------------------------
# Data + config
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Kendall County parcels…")
def load_index() -> E.ParcelIndex:
    return E.ParcelIndex.from_parquet(PARQUET)


def sidebar_config() -> dict:
    st.sidebar.header("Layout assumptions")
    st.sidebar.caption("The parcel is developed from a blank slate using these "
                       "program and engineering assumptions.")

    cfg = {}
    cfg["target_density_du_ac"] = st.sidebar.slider(
        "Target density (du/ac)", 2, 30, 10)
    cfg["units_per_building"] = st.sidebar.slider(
        "Units per building", 1, 24, 6)
    c1, c2 = st.sidebar.columns(2)
    cfg["building_width_ft"] = c1.number_input("Building width (ft)", 20, 300, 80, step=5)
    cfg["building_depth_ft"] = c2.number_input("Building depth (ft)", 20, 200, 42, step=2)
    cfg["road_spacing_ft"] = st.sidebar.slider("Road spacing (ft)", 80, 400, 180, step=10)
    cfg["road_width_ft"] = st.sidebar.slider("Road width (ft)", 18, 40, 26, step=2)
    cfg["edge_buffer_ft"] = st.sidebar.slider("Perimeter setback (ft)", 0, 120, 40, step=5)

    st.sidebar.header("Terrain")
    cfg["max_buildable_slope_pct"] = st.sidebar.slider(
        "Max buildable slope (%)", 5, 30, 15,
        help="Ground steeper than this is excluded from the developable core.")
    cfg["pond_pct"] = st.sidebar.slider(
        "Stormwater pond (% of net area)", 2, 20, 10, step=1) / 100.0
    cfg["terrain_resolution_m"] = st.sidebar.select_slider(
        "3DEP resolution (m)", options=[1, 3, 10], value=3,
        help="Finer is slower. 1 m for small sites, 10 m for very large ones.")
    cfg["force_synthetic_terrain"] = st.sidebar.checkbox(
        "Force synthetic terrain (offline demo)", value=False)

    return cfg


# ---------------------------------------------------------------------------
# Parcel resolution
# ---------------------------------------------------------------------------
def resolve_by_address(idx: E.ParcelIndex, query: str):
    """Return (list_of_candidate_rows, note). Try local substring search first
    (fast, offline, exact to the parcel dataset); fall back to geocoding for
    addresses not carried in the parcel table."""
    hits = idx.search_address(query)
    if len(hits):
        return [hits.iloc[i] for i in range(len(hits))], None
    ll = E.geocode_address(query)
    if ll is not None:
        row = idx.by_point(*ll)
        if row is not None:
            return [row], f"Matched by geocoding → nearest parcel ({row['full_address']})."
    return [], "No parcel found for that address. Try a street name, or click the map."


def select_parcel(row):
    """Store the chosen parcel in session state (keyed for cache invalidation)."""
    st.session_state["parcel_geom"] = row.geometry
    st.session_state["parcel_meta"] = {
        "address": row.get("address", ""),
        "city": row.get("city", ""),
        "parcelnumb": row.get("parcelnumb", ""),
        "full_address": row.get("full_address", ""),
        "gisacre": row.get("gisacre", None),
        "lat": row.get("lat", None),
        "lon": row.get("lon", None),
    }
    st.session_state.pop("plan_cache", None)


# ---------------------------------------------------------------------------
# Plan computation (recompute only when parcel or config changes)
# ---------------------------------------------------------------------------
def get_plan(config: dict):
    meta = st.session_state["parcel_meta"]
    geom = st.session_state["parcel_geom"]
    sig = (meta.get("parcelnumb"), tuple(sorted(config.items())))
    cached = st.session_state.get("plan_cache")
    if cached and cached[0] == sig:
        return cached[1]
    plan = E.plan_for_geometry(
        geom, meta.get("address", ""), meta.get("city", ""),
        meta.get("parcelnumb", ""), config_overrides=config)
    st.session_state["plan_cache"] = (sig, plan)
    return plan


# ---------------------------------------------------------------------------
# UI sections
# ---------------------------------------------------------------------------
def render_locator(meta):
    """Small folium map showing where the parcel sits, with its outline."""
    center = (meta["lat"], meta["lon"]) if meta.get("lat") else COUNTY_CENTER
    m = folium.Map(location=center, zoom_start=16, tiles="OpenStreetMap",
                   control_scale=True)
    geom = st.session_state["parcel_geom"]
    folium.GeoJson(mapping(geom),
                   style_function=lambda _: {"color": "#d62728", "weight": 3,
                                             "fillOpacity": 0.15}).add_to(m)
    folium.Marker(center, tooltip=meta.get("full_address") or "Parcel").add_to(m)
    st_folium(m, height=360, use_container_width=True, key="locator")


def render_metrics(m):
    if m["too_small"]:
        st.warning(
            f"This parcel is **{m['gross_acres']:.2f} acres** — after the "
            f"{st.session_state.get('edge_buffer_ft','')} perimeter setback and "
            "constraints there is no developable core for the current building "
            "program. Try a larger parcel, a smaller setback, or a smaller "
            "building footprint in the sidebar.")

    a, b, c, d = st.columns(4)
    a.metric("Gross acres", f"{m['gross_acres']:.2f}")
    b.metric("Net developable", f"{m['net_acres']:.2f} ac")
    c.metric("Buildings", f"{m['buildings']:,}")
    d.metric("Units", f"{m['units']:,}")

    a, b, c, d = st.columns(4)
    a.metric("Gross density", f"{m['gross_density']:.1f} du/ac",
             f"{m['pct_of_target']:.0f}% of target ({m['target_density']})")
    b.metric("Building coverage", f"{m['coverage_pct']:.1f}%")
    c.metric("Grading risk", m["grading_risk"],
             f"{m['earthwork_intensity']:,.0f} cy/ac")
    d.metric("Earthwork cost", f"${m['earthwork_cost']:,.0f}")

    with st.expander("Terrain & earthwork detail"):
        src = ("real USGS 3DEP elevation" if m["terrain_source"] == "3DEP"
               else "synthetic fallback surface (no live 3DEP)")
        st.write(f"**Terrain source:** {src}")
        st.write(f"**Mean elevation:** {m['mean_elevation_ft']:.0f} ft &nbsp;•&nbsp; "
                 f"**Best-fit grade:** {m['plane_slope_pct']:.2f}%")
        st.write(f"**Layout orientation:** {m['orientation']}")
        st.write(f"**Cut:** {m['cut_cy']:,.0f} cy &nbsp;•&nbsp; "
                 f"**Fill:** {m['fill_cy']:,.0f} cy &nbsp;•&nbsp; "
                 f"**Net:** {abs(m['net_cy']):,.0f} cy "
                 f"({'export' if m['net_export'] else 'import'})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    idx = load_index()
    config = sidebar_config()
    st.session_state["edge_buffer_ft"] = config["edge_buffer_ft"]

    st.title("🏗️ Kendall County Terrain-Informed Site Planner")
    st.caption("Enter any address in Kendall County, Illinois — or click the map — "
               "and the parcel is developed from a blank slate using its real "
               "USGS 3DEP terrain.")

    left, right = st.columns([1, 1], gap="large")

    # ---- Input: address search ----
    with left:
        st.subheader("1 · Find a parcel")
        with st.form("addr_form"):
            query = st.text_input("Address or street name",
                                  placeholder="e.g. 309 Danforth Dr, or Fox Rd")
            submitted = st.form_submit_button("Search", width="stretch")
        if submitted and query.strip():
            cands, note = resolve_by_address(idx, query)
            st.session_state["candidates"] = [
                {"label": (r.get("full_address") or r.get("parcelnumb")),
                 "idx": int(r.name)} for r in cands]
            if note:
                st.info(note)
            if not cands:
                st.session_state.pop("candidates", None)

        cands = st.session_state.get("candidates")
        if cands:
            labels = [c["label"] for c in cands]
            pick = st.selectbox(f"{len(cands)} match(es) — choose one", labels)
            if st.button("Develop this parcel", type="primary",
                         width="stretch"):
                chosen = next(c for c in cands if c["label"] == pick)
                select_parcel(idx.gdf.iloc[chosen["idx"]])

    # ---- Input: map click ----
    with right:
        st.subheader("2 · …or click the map")
        m = folium.Map(location=COUNTY_CENTER, zoom_start=11,
                       tiles="OpenStreetMap", control_scale=True)
        m.fit_bounds(COUNTY_BOUNDS)
        folium.Rectangle(COUNTY_BOUNDS, color="#1f77b4", weight=1,
                         fill=False, tooltip="Kendall County (approx.)").add_to(m)
        out = st_folium(m, height=360, use_container_width=True, key="picker")
        clicked = out.get("last_clicked") if out else None
        if clicked:
            row = idx.by_point(clicked["lng"], clicked["lat"])
            if row is not None:
                st.success(f"Selected: {row.get('full_address') or row.get('parcelnumb')}")
                if st.button("Develop clicked parcel", type="primary",
                             width="stretch", key="dev_click"):
                    select_parcel(row)

    # ---- Results ----
    if "parcel_geom" in st.session_state:
        meta = st.session_state["parcel_meta"]
        st.divider()
        title = meta.get("full_address") or meta.get("parcelnumb") or "Selected parcel"
        st.subheader(f"Site plan · {title}")
        if meta.get("parcelnumb"):
            st.caption(f"Parcel {meta['parcelnumb']}"
                       + (f" · {meta['gisacre']:.2f} county-GIS acres"
                          if meta.get("gisacre") else ""))

        with st.spinner("Pulling terrain and developing the parcel…"):
            plan = get_plan(config)

        render_metrics(plan.metrics)

        col_plan, col_map = st.columns([3, 2], gap="large")
        with col_plan:
            fig = E.render_plan(plan)
            st.pyplot(fig, width="stretch")
        with col_map:
            render_locator(meta)
    else:
        st.info("Search an address or click the map, then choose a parcel to develop.")


if __name__ == "__main__":
    main()
