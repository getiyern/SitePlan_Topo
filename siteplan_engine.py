"""
siteplan_engine.py -- terrain-informed, blank-slate site-plan engine for
Kendall County, Illinois parcels.

This is the notebook `SitePlan_TerrainInformed_1.ipynb` refactored into a clean,
importable library so a UI (see app.py) can call it per address. The pipeline,
end to end, for one parcel:

  1. Locate the parcel   -- by map point (point-in-polygon) or address search.
  2. Reproject to feet   -- NAD83 / Illinois East (ftUS), recenter to origin.
  3. Acquire terrain     -- real USGS 3DEP DEM for the parcel footprint, with a
                            synthetic sloped-surface fallback when offline.
  4. Analyze terrain     -- best-fit grading plane, local slope, steep-slope
                            exclusion zone, low-point sampler.
  5. Lay out from scratch-- edge buffer, constraints, terrain-seeking pond,
                            contour-aligned roads and buildings, density, and
                            real cut/fill earthwork economics.

Everything the notebook kept in module-level globals (dem_ft, cellsize_ft,
residual grid, ...) is carried on a `Terrain` object instead, so the engine is
re-entrant and safe to call many times in one process.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon, Point, box, LineString, shape as shp_shape
from shapely.affinity import translate, rotate as shp_rotate
from shapely.ops import unary_union
from shapely.strtree import STRtree

warnings.filterwarnings("ignore")

SQFT_PER_ACRE = 43560
WGS84 = "EPSG:4326"
# NAD83 / Illinois East (ftUS) -- the correct State Plane zone for Kendall County.
PROJECTED_CRS = "EPSG:3435"


# ---------------------------------------------------------------------------
# Default layout / economics configuration. The UI can override any of these.
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "projected_crs": PROJECTED_CRS,
    "aspect_ratio": 1.4,

    # constraints
    "constraints_enabled": True,
    "wetland_pct": 0.06, "floodplain_pct": 0.02,
    "edge_buffer_ft": 40, "wetland_buffer_ft": 5,

    # terrain-driven constraints
    "use_slope_exclusion": True,
    "max_buildable_slope_pct": 15.0,

    # program geometry
    "pond_pct": 0.10,
    "road_width_ft": 26, "road_spacing_ft": 180,
    "building_width_ft": 80, "building_depth_ft": 42,
    "building_spacing_ft": 12, "rear_separation_ft": 30,
    "units_per_building": 6,
    "auto_orient_buildings": True,
    "grid_search_steps": 4,

    # terrain-driven orientation
    "orient_to_terrain": True,
    "terrain_orientation_min_slope_pct": 2.0,

    # targets & economics
    "target_density_du_ac": 10,
    "cut_cost_per_cy": 9.0,
    "fill_cost_per_cy": 12.0,

    # terrain acquisition
    "terrain_resolution_m": 3,
    "terrain_buffer_ft": 100,
    "force_synthetic_terrain": False,
}


def merged_config(overrides: Optional[dict] = None) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if overrides:
        cfg.update(overrides)
    return cfg


# ===========================================================================
# 1. Parcel index -- load once, query by point or address
# ===========================================================================
class ParcelIndex:
    """In-memory parcel store with an STRtree for fast point-in-polygon lookup."""

    def __init__(self, gdf: gpd.GeoDataFrame):
        if gdf.crs is None:
            gdf = gdf.set_crs(WGS84)
        elif str(gdf.crs).upper() not in ("EPSG:4326",):
            gdf = gdf.to_crs(WGS84)
        self.gdf = gdf.reset_index(drop=True)
        self._geoms = list(self.gdf.geometry.values)
        self._tree = STRtree(self._geoms)
        # normalized address for case-insensitive substring search
        self._addr_norm = self.gdf["address"].fillna("").str.upper()

    @classmethod
    def from_parquet(cls, path: str) -> "ParcelIndex":
        return cls(gpd.read_parquet(path))

    def by_point(self, lon: float, lat: float):
        """Return the parcel row whose polygon contains (lon, lat), or None.

        Falls back to the nearest parcel centroid when the point lands in a gap
        between polygons (road right-of-way, water) so a map click always
        resolves to *some* parcel.
        """
        pt = Point(lon, lat)
        for idx in self._tree.query(pt):
            geom = self._geoms[int(idx)]
            if geom.contains(pt):
                return self.gdf.iloc[int(idx)]
        # nearest fallback
        try:
            nidx = self._tree.nearest(pt)
            return self.gdf.iloc[int(nidx)]
        except Exception:
            return None

    def search_address(self, query: str, limit: int = 25) -> pd.DataFrame:
        """Case-insensitive substring match on the address, best matches first."""
        q = (query or "").strip().upper()
        if not q:
            return self.gdf.iloc[0:0]
        mask = self._addr_norm.str.contains(q, regex=False, na=False)
        hits = self.gdf[mask]
        if hits.empty:
            return hits
        # rank: exact-prefix first, then shorter (closer) addresses
        order = hits["address"].str.upper().map(
            lambda a: (0 if a.startswith(q) else 1, len(a)))
        return hits.loc[order.sort_values().index].head(limit)


def geocode_address(address: str, county_bias: str = "Kendall County, Illinois, USA"):
    """Geocode a free-text address to (lon, lat) using OpenStreetMap Nominatim.

    Returns None if geocoding is unavailable (offline) or finds nothing. The app
    can still fall back to `ParcelIndex.search_address`.
    """
    try:
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(user_agent="kendall_siteplan", timeout=10)
        for q in (f"{address}, {county_bias}", address):
            loc = geolocator.geocode(q)
            if loc is not None:
                return (loc.longitude, loc.latitude)
    except Exception:
        pass
    return None


# ===========================================================================
# 2. Prepare parcel geometry (reproject to feet, recenter)
# ===========================================================================
def largest_polygon(geom):
    """Largest single Polygon (collapses a MultiPolygon parcel to its main ring)."""
    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda g: g.area)
    return geom


@dataclass
class Parcel:
    polygon_ft: Polygon          # recentered to origin, in projected feet
    centroid_ft: tuple           # (cx, cy) in the projected CRS, pre-recenter
    bounds_wgs84: tuple          # (minx, miny, maxx, maxy) lon/lat
    acres: float
    address: str
    city: str
    parcelnumb: str


def prepare_parcel(geom_wgs84, address="", city="", parcelnumb="",
                   projected_crs=PROJECTED_CRS) -> Parcel:
    """Reproject a WGS84 parcel geometry to feet and recenter its centroid to (0,0)."""
    gs = gpd.GeoSeries([geom_wgs84], crs=WGS84)
    bounds_wgs84 = tuple(gs.total_bounds)
    geom_ft = largest_polygon(gs.to_crs(projected_crs).iloc[0])
    cx, cy = geom_ft.centroid.x, geom_ft.centroid.y
    polygon_ft = translate(geom_ft, xoff=-cx, yoff=-cy)
    acres = polygon_ft.area / SQFT_PER_ACRE
    return Parcel(polygon_ft, (cx, cy), bounds_wgs84, acres,
                  address or "", city or "", parcelnumb or "")


# ===========================================================================
# 3-4. Terrain acquisition + analysis
# ===========================================================================
@dataclass
class Terrain:
    dem_ft: np.ndarray
    cellsize_ft: float
    bounds_ft: tuple             # (minx, miny, maxx, maxy) recentered feet
    source: str                  # "3DEP" or "synthetic"
    mean_elevation_ft: float = 0.0
    plane_slope_pct: float = 0.0
    downhill_bearing_deg: float = 0.0
    residual_from_plane_ft: np.ndarray = None
    local_slope_pct: np.ndarray = None
    steep_slope_zone: object = None

    @property
    def rows(self):
        return self.dem_ft.shape[0]

    @property
    def cols(self):
        return self.dem_ft.shape[1]

    def dem_value_at(self, px, py):
        """Nearest-neighbor DEM lookup in the recentered feet coordinate space."""
        minx, miny, maxx, maxy = self.bounds_ft
        col = int(np.clip((px - minx) / self.cellsize_ft, 0, self.cols - 1))
        row = int(np.clip((maxy - py) / self.cellsize_ft, 0, self.rows - 1))
        return self.dem_ft[row, col]

    def real_cut_fill(self, footprint_poly):
        """Cut/fill (cy) for a footprint: residual vs. the best-fit grading plane,
        with 25% swell/compaction, exactly as the topo notebook computes it."""
        if footprint_poly is None or footprint_poly.is_empty or footprint_poly.area <= 0:
            return 0.0, 0.0, 0.0
        minx, miny, maxx, maxy = self.bounds_ft
        fminx, fminy, fmaxx, fmaxy = footprint_poly.bounds
        col0 = int(np.clip((fminx - minx) / self.cellsize_ft, 0, self.cols - 1))
        col1 = int(np.clip((fmaxx - minx) / self.cellsize_ft, 0, self.cols - 1))
        row0 = int(np.clip((maxy - fmaxy) / self.cellsize_ft, 0, self.rows - 1))
        row1 = int(np.clip((maxy - fminy) / self.cellsize_ft, 0, self.rows - 1))
        if col1 <= col0 or row1 <= row0:
            return 0.0, 0.0, 0.0
        sub = self.residual_from_plane_ft[row0:row1, col0:col1]
        cell_area = self.cellsize_ft ** 2
        cut_cy = float(np.sum(sub[sub > 0]) * cell_area / 27.0)
        fill_cy = float(np.sum(np.abs(sub[sub < 0])) * cell_area / 27.0)
        swell, compact = 0.25, 0.25
        net_cy = cut_cy * (1 + swell) - fill_cy * (1 + compact)
        return cut_cy, fill_cy, net_cy


def _fetch_synthetic_dem(bounds, resolution_ft=10, slope_pct=3.0, aspect_deg=225,
                         noise_std=0.4, seed=0):
    """Planar grade + noise standing in for a real DEM when 3DEP is unavailable."""
    minx, miny, maxx, maxy = bounds
    nx = max(20, int((maxx - minx) / resolution_ft))
    ny = max(20, int((maxy - miny) / resolution_ft))
    x = np.linspace(minx, maxx, nx)
    y = np.linspace(miny, maxy, ny)
    X, Y = np.meshgrid(x, y)
    rad = np.deg2rad(aspect_deg)
    grade = slope_pct / 100.0
    Z = 700 + grade * (X * np.cos(rad) + Y * np.sin(rad))
    rng = np.random.default_rng(seed)
    Z += rng.normal(0, noise_std, Z.shape)
    return Z[::-1, :], resolution_ft, bounds


def acquire_terrain(parcel: Parcel, config: dict) -> Terrain:
    """Pull a USGS 3DEP DEM for the parcel footprint (reprojected to the site's
    feet CRS and recentered), or fall back to a synthetic sloped surface."""
    cx, cy = parcel.centroid_ft
    buffer_ft = config["terrain_buffer_ft"]
    force_synth = config.get("force_synthetic_terrain", False)

    minx_ll, miny_ll, maxx_ll, maxy_ll = parcel.bounds_wgs84
    pad_deg = buffer_ft / 364000.0
    dem_bbox_wgs84 = box(minx_ll - pad_deg, miny_ll - pad_deg,
                         maxx_ll + pad_deg, maxy_ll + pad_deg)

    dem_ft = cellsize_ft = bounds_ft = None
    source = "synthetic"

    if not force_synth:
        try:
            import py3dep  # noqa: F401  (import side effects only)
            grid = py3dep.get_map(geometry=dem_bbox_wgs84,
                                  resolution=config["terrain_resolution_m"],
                                  crs="epsg:4326", layers="DEM")
            grid_ft = grid.rio.reproject(config["projected_crs"])
            raw = grid_ft.values.squeeze().astype(float)
            nodata = grid_ft.rio.nodata
            if nodata is not None:
                raw = np.where(raw == nodata, np.nan, raw)
            dem_ft = raw * 3.28084  # meters -> feet
            t = grid_ft.rio.transform()
            cellsize_ft = abs(t.a)
            b = grid_ft.rio.bounds()
            bounds_ft = (b[0] - cx, b[1] - cy, b[2] - cx, b[3] - cy)
            source = "3DEP"
        except Exception:
            dem_ft = None

    if dem_ft is None:
        minx, miny, maxx, maxy = parcel.polygon_ft.bounds
        pad = buffer_ft
        dem_ft, cellsize_ft, bounds_ft = _fetch_synthetic_dem(
            (minx - pad, miny - pad, maxx + pad, maxy + pad), resolution_ft=10)
        source = "synthetic"

    if np.isnan(dem_ft).any():
        dem_ft = np.where(np.isnan(dem_ft), np.nanmean(dem_ft), dem_ft)

    terrain = Terrain(dem_ft=dem_ft, cellsize_ft=cellsize_ft,
                      bounds_ft=bounds_ft, source=source)
    _analyze_terrain(terrain, config)
    return terrain


def _analyze_terrain(t: Terrain, config: dict) -> None:
    """Best-fit grading plane, residual/slope grids, and steep-slope exclusion zone."""
    minx, miny, maxx, maxy = t.bounds_ft
    rows, cols = t.dem_ft.shape
    xs = np.linspace(minx, maxx, cols)
    ys = np.linspace(maxy, miny, rows)   # row 0 = top = maxy
    Xg, Yg = np.meshgrid(xs, ys)

    t.mean_elevation_ft = float(np.nanmean(t.dem_ft))

    A_mat = np.c_[Xg.ravel(), Yg.ravel(), np.ones(Xg.size)]
    coeffs, *_ = np.linalg.lstsq(A_mat, t.dem_ft.ravel(), rcond=None)
    gx, gy, gc = coeffs
    fitted = gx * Xg + gy * Yg + gc

    t.residual_from_plane_ft = t.dem_ft - fitted
    t.plane_slope_pct = float(np.hypot(gx, gy) * 100)
    t.downhill_bearing_deg = float(np.degrees(np.arctan2(-gy, -gx)) % 360)

    dzdy, dzdx = np.gradient(t.dem_ft, t.cellsize_ft)
    t.local_slope_pct = np.hypot(dzdx, dzdy) * 100

    # Vectorize steep cells into a real exclusion polygon (recentered feet CRS).
    max_slope = config["max_buildable_slope_pct"]
    steep_mask = t.local_slope_pct > max_slope
    steep_zone = None
    if steep_mask.any():
        try:
            import rasterio.features
            from affine import Affine
            transform = Affine(t.cellsize_ft, 0, minx, 0, -t.cellsize_ft, maxy)
            shapes_gen = rasterio.features.shapes(
                steep_mask.astype(np.uint8), mask=steep_mask, transform=transform)
            polys = [shp_shape(g) for g, v in shapes_gen if v == 1]
            if polys:
                steep_zone = unary_union(polys).buffer(0)
        except Exception:
            steep_zone = None
    t.steep_slope_zone = steep_zone


# ===========================================================================
# 5. Blank-slate Tier 4 layout
# ===========================================================================
def _valid_poly(poly):
    """True if the geometry is a usable, non-empty polygon with real extent."""
    if poly is None or getattr(poly, "is_empty", True):
        return False
    minx, miny, maxx, maxy = poly.bounds
    return (maxx > minx) and (maxy > miny) and poly.area > 0


def _long_axis_is_y(poly):
    minx, miny, maxx, maxy = poly.bounds
    return (maxy - miny) >= (maxx - minx)


def create_developable_area(site_poly, config):
    return largest_polygon(site_poly.buffer(-config["edge_buffer_ft"]))


def simulate_constraints(dev_poly, config, slope_zone=None):
    """Subtract the real DEM steep-slope zone plus synthetic wetland/floodplain
    placeholders from the developable core."""
    if not config.get("constraints_enabled", True):
        return dev_poly, None, None, None
    minx, miny, maxx, maxy = dev_poly.bounds
    area = dev_poly.area
    wetland = Point(minx + (maxx - minx) * 0.18, miny + (maxy - miny) * 0.85).buffer(
        np.sqrt(area * config["wetland_pct"] / np.pi))
    flood = Point(minx + (maxx - minx) * 0.85, miny + (maxy - miny) * 0.15).buffer(
        np.sqrt(area * config["floodplain_pct"] / np.pi))
    exclusions = [wetland.buffer(config["wetland_buffer_ft"]), flood]

    slope_excl = None
    if config.get("use_slope_exclusion") and slope_zone is not None and not slope_zone.is_empty:
        candidate = slope_zone.intersection(dev_poly)
        if not candidate.is_empty:
            slope_excl = candidate
            exclusions.append(slope_excl)

    net = dev_poly.difference(unary_union(exclusions))
    return largest_polygon(net), wetland, flood, slope_excl


def pick_pond_site(net_poly, config, dem_sampler=None):
    """Anchor the stormwater pond at the lowest DEM elevation inside the core."""
    from shapely.geometry import Polygon as _Poly
    if not _valid_poly(net_poly):
        return _Poly()
    radius = np.sqrt(net_poly.area * config["pond_pct"] / np.pi)
    minx, miny, maxx, maxy = net_poly.bounds
    if dem_sampler is not None and maxx - radius > minx + radius and maxy - radius > miny + radius:
        gx = np.linspace(minx + radius, maxx - radius, 12)
        gy = np.linspace(miny + radius, maxy - radius, 12)
        best_pt, best_z = None, np.inf
        for px in gx:
            for py in gy:
                pt = Point(px, py)
                if net_poly.contains(pt):
                    z = dem_sampler(px, py)
                    if z < best_z:
                        best_z, best_pt = z, pt
        if best_pt is not None:
            return best_pt.buffer(radius)
    return Point(minx + radius * 0.7, miny + radius * 0.7).buffer(radius)


def terrain_orientation_bearing(config, plane_slope_pct, downhill_bearing_deg):
    """Contour bearing (perpendicular to downhill) once the grade is steep enough."""
    if (config.get("orient_to_terrain")
            and plane_slope_pct >= config.get("terrain_orientation_min_slope_pct", 2.0)):
        return (downhill_bearing_deg + 90) % 180
    return None


def generate_roads(net_poly, pond, config, orientation_deg=None):
    if not _valid_poly(net_poly):
        return []
    minx, miny, maxx, maxy = net_poly.bounds
    spacing, width = config["road_spacing_ft"], config["road_width_ft"]

    if orientation_deg is not None:
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        diag = np.hypot(maxx - minx, maxy - miny)
        theta = np.deg2rad(orientation_deg)
        dirx, diry = np.cos(theta), np.sin(theta)
        nx, ny = -diry, dirx
        roads = []
        for s in np.arange(-diag / 2, diag / 2, spacing):
            ox, oy = cx + nx * s, cy + ny * s
            seg = LineString([(ox - dirx * diag, oy - diry * diag),
                              (ox + dirx * diag, oy + diry * diag)])
            r = largest_polygon(net_poly.intersection(
                seg.buffer(width / 2, cap_style=3, join_style=3)))
            if (not r.is_empty) and (not r.intersects(pond)) and r.area > width * 40:
                roads.append(r)
        return roads

    if _long_axis_is_y(net_poly):
        coords = np.arange(miny + spacing, maxy - spacing / 2, spacing)
        make = lambda c: LineString([(minx, c), (maxx, c)])
    else:
        coords = np.arange(minx + spacing, maxx - spacing / 2, spacing)
        make = lambda c: LineString([(c, miny), (c, maxy)])
    roads = []
    for c in coords:
        seg = make(c).buffer(width / 2, cap_style=3, join_style=3)
        r = largest_polygon(net_poly.intersection(seg))
        if (not r.is_empty) and (not r.intersects(pond)) and r.area > width * 40:
            roads.append(r)
    return roads


def place_buildings(net_poly, roads, pond, config, orientation_deg=None):
    if not _valid_poly(net_poly):
        return []
    b_w, b_d = config["building_width_ft"], config["building_depth_ft"]
    use_terrain_axis = orientation_deg is not None
    if (config.get("auto_orient_buildings", True) and not use_terrain_axis
            and _long_axis_is_y(net_poly)):
        fw, fd = b_d, b_w
    else:
        fw, fd = b_w, b_d

    side, row_gap = config["building_spacing_ft"], config["rear_separation_ft"]
    blockers = [g for g in ([pond] + list(roads)) if g is not None]
    blocked = unary_union(blockers) if blockers else None
    steps = max(1, int(config.get("grid_search_steps", 4)))

    if not use_terrain_axis:
        minx, miny, maxx, maxy = net_poly.bounds
        best = []
        for ox in np.linspace(0, fw + side, steps):
            for oy in np.linspace(0, fd + row_gap, steps):
                placed = []
                y = miny + oy + 5
                while y < maxy - fd:
                    x = minx + ox + 5
                    while x < maxx - fw:
                        b = box(x, y, x + fw, y + fd)
                        if net_poly.contains(b) and (blocked is None or not b.intersects(blocked)):
                            placed.append(b)
                        x += fw + side
                    y += fd + row_gap
                if len(placed) > len(best):
                    best = placed
        return best

    theta = orientation_deg
    cx, cy = net_poly.centroid.x, net_poly.centroid.y
    net_rot = shp_rotate(net_poly, -theta, origin=(cx, cy))
    blocked_rot = shp_rotate(blocked, -theta, origin=(cx, cy)) if blocked is not None else None
    rminx, rminy, rmaxx, rmaxy = net_rot.bounds

    best = []
    for ox in np.linspace(0, fw + side, steps):
        for oy in np.linspace(0, fd + row_gap, steps):
            placed = []
            y = rminy + oy + 5
            while y < rmaxy - fd:
                x = rminx + ox + 5
                while x < rmaxx - fw:
                    b = box(x, y, x + fw, y + fd)
                    if net_rot.contains(b) and (blocked_rot is None or not b.intersects(blocked_rot)):
                        placed.append(b)
                    x += fw + side
                y += fd + row_gap
            if len(placed) > len(best):
                best = placed
    return [shp_rotate(b, theta, origin=(cx, cy)) for b in best]


def grading_risk(cut_cy, fill_cy, net_acres):
    intensity = (cut_cy + fill_cy) / net_acres if net_acres else 0
    if intensity < 400:
        return "Low", intensity
    if intensity < 1500:
        return "Moderate", intensity
    return "High", intensity


@dataclass
class SitePlan:
    parcel: Parcel
    terrain: Terrain
    config: dict
    site: Polygon
    net_area: Polygon
    wetlands: object
    floodplain: object
    slope_excl: object
    pond: object
    roads: list
    buildings: list
    orientation_deg: Optional[float]
    metrics: dict = field(default_factory=dict)


def generate_site_plan(parcel: Parcel, terrain: Terrain, config: dict) -> SitePlan:
    """Develop the parcel from a blank slate and compute underwriting metrics."""
    site = largest_polygon(parcel.polygon_ft)
    developable = create_developable_area(site, config)
    net_area, wetlands, floodplain, slope_excl = simulate_constraints(
        developable, config, slope_zone=terrain.steep_slope_zone)

    orientation_deg = terrain_orientation_bearing(
        config, terrain.plane_slope_pct, terrain.downhill_bearing_deg)

    # A parcel smaller than one building footprint (after the edge buffer and
    # constraints) has no developable core -- return an empty program cleanly
    # instead of forcing geometry through the layout engine.
    from shapely.geometry import Polygon as _Poly
    min_footprint = config["building_width_ft"] * config["building_depth_ft"]
    if _valid_poly(net_area) and net_area.area >= min_footprint:
        pond = pick_pond_site(net_area, config, dem_sampler=terrain.dem_value_at)
        roads = generate_roads(net_area, pond, config, orientation_deg)
        buildings = place_buildings(net_area, roads, pond, config, orientation_deg)
        too_small = False
    else:
        pond, roads, buildings = _Poly(), [], []
        too_small = True

    gross_acres = site.area / SQFT_PER_ACRE
    net_acres = (net_area.area / SQFT_PER_ACRE) if _valid_poly(net_area) else 0.0
    unit_count = len(buildings) * config["units_per_building"]
    gross_density = unit_count / gross_acres if gross_acres else 0
    net_density = unit_count / net_acres if net_acres else 0
    building_sqft = sum(b.area for b in buildings)
    coverage_pct = building_sqft / site.area * 100 if site.area else 0
    target = config["target_density_du_ac"]

    cut_cy, fill_cy, net_cy = terrain.real_cut_fill(net_area)
    risk, intensity = grading_risk(cut_cy, fill_cy, net_acres)
    earthwork_cost = cut_cy * config["cut_cost_per_cy"] + fill_cy * config["fill_cost_per_cy"]

    metrics = {
        "address": parcel.address,
        "city": parcel.city,
        "parcelnumb": parcel.parcelnumb,
        "terrain_source": terrain.source,
        "gross_acres": gross_acres,
        "net_acres": net_acres,
        "too_small": too_small,
        "buildings": len(buildings),
        "units": unit_count,
        "units_per_building": config["units_per_building"],
        "coverage_pct": coverage_pct,
        "gross_density": gross_density,
        "net_density": net_density,
        "target_density": target,
        "pct_of_target": (gross_density / target * 100) if target else 0,
        "road_count": len(roads),
        "orientation": (f"terrain contour ({orientation_deg:.0f}° axis)"
                        if orientation_deg is not None
                        else "geometric long axis (flat/low-slope site)"),
        "mean_elevation_ft": terrain.mean_elevation_ft,
        "plane_slope_pct": terrain.plane_slope_pct,
        "cut_cy": cut_cy,
        "fill_cy": fill_cy,
        "net_cy": net_cy,
        "net_export": net_cy > 0,
        "earthwork_cost": earthwork_cost,
        "earthwork_intensity": intensity,
        "grading_risk": risk,
    }

    return SitePlan(parcel, terrain, config, site, net_area, wetlands, floodplain,
                    slope_excl, pond, roads, buildings, orientation_deg, metrics)


# ===========================================================================
# 6. Rendering
# ===========================================================================
def _fill_geom(ax, geom, label=None, **kw):
    if geom is None or getattr(geom, "is_empty", True):
        return
    parts = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    first = True
    for part in parts:
        gx, gy = part.exterior.xy
        ax.fill(gx, gy, label=(label if first else None), **kw)
        first = False


def render_plan(plan: SitePlan, figsize=(9, 11)):
    """Matplotlib figure of the finished site plan (returns the Figure)."""
    import matplotlib.pyplot as plt
    m = plan.metrics
    fig, ax = plt.subplots(figsize=figsize)
    edge = plt.rcParams.get("text.color", "black")

    bx, by = plan.site.exterior.xy
    ax.plot(bx, by, linewidth=2, color=edge, label="Parcel Boundary")

    _fill_geom(ax, plan.net_area, label="Developable Area", alpha=0.2)
    _fill_geom(ax, plan.wetlands, label="Wetlands", alpha=0.5)
    _fill_geom(ax, plan.floodplain, label="Floodplain", alpha=0.5)
    _fill_geom(ax, plan.slope_excl,
               label=f"Steep Slope (>{plan.config['max_buildable_slope_pct']:.0f}%)",
               alpha=0.5, color="saddlebrown")
    _fill_geom(ax, plan.pond, label="Stormwater Pond", alpha=0.7)

    first = True
    for road in plan.roads:
        _fill_geom(ax, road, label=("Private Roads" if first else None), alpha=0.6)
        first = False
    first = True
    for b in plan.buildings:
        _fill_geom(ax, b, label=("Buildings" if first else None), alpha=0.85)
        first = False

    title_addr = m["address"] or plan.parcel.parcelnumb or "Selected parcel"
    ax.set_title(
        f"Terrain-Informed Site Plan -- {title_addr}\n"
        f"{m['buildings']} buildings | {m['units']} units | "
        f"{m['gross_density']:.1f} du/ac | {m['coverage_pct']:.0f}% coverage | "
        f"Grading Risk: {m['grading_risk']}", fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlabel("Feet (E-W, recentered)")
    ax.set_ylabel("Feet (N-S, recentered)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.4)
    fig.tight_layout()
    return fig


# ===========================================================================
# Convenience one-shot
# ===========================================================================
def plan_for_geometry(geom_wgs84, address="", city="", parcelnumb="",
                      config_overrides=None) -> SitePlan:
    """Full pipeline for a single WGS84 parcel geometry."""
    config = merged_config(config_overrides)
    parcel = prepare_parcel(geom_wgs84, address, city, parcelnumb,
                            projected_crs=config["projected_crs"])
    terrain = acquire_terrain(parcel, config)
    return generate_site_plan(parcel, terrain, config)
