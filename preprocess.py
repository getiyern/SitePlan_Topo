"""
preprocess.py -- one-time conversion of the raw Kendall County parcel GeoJSON
into a compact, fast-loading GeoParquet used by the web app.

The raw file (~325 MB) is far too slow to parse on every app run. This script
loads it once, normalizes an address string for each parcel (the raw `address`
field is null for ~12% of parcels, so we fall back to the component fields),
keeps only the columns the app needs, and writes `parcels.parquet`.

Run once:
    python preprocess.py
Optionally point at a different source/destination:
    python preprocess.py --src "C:/path/il_kendall.json" --dst parcels.parquet
"""
from __future__ import annotations

import argparse
import json
import os

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape

DEFAULT_SRC = "C:/Users/neeli/OneDrive/Documents/Heyday/il_kendall.geojson/il_kendall.json"
DEFAULT_DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parcels.parquet")

WGS84 = "EPSG:4326"


def _nn(v) -> bool:
    """True if a property value is present and meaningful."""
    return v not in (None, "", "null", "None")


def compose_address(pr: dict) -> str:
    """Best available street address for a parcel.

    Prefer the raw `address` field; otherwise assemble it from the street
    component fields (saddno / saddpref / saddstr / saddsttyp / saddstsuf).
    """
    if _nn(pr.get("address")):
        return str(pr["address"]).strip()
    parts = [
        pr.get("saddno"), pr.get("saddpref"), pr.get("saddstr"),
        pr.get("saddsttyp"), pr.get("saddstsuf"),
    ]
    return " ".join(str(p).strip() for p in parts if _nn(p))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=DEFAULT_SRC, help="Raw parcel GeoJSON/JSON path")
    ap.add_argument("--dst", default=DEFAULT_DST, help="Output GeoParquet path")
    args = ap.parse_args()

    print(f"Loading raw parcels from {args.src} ...")
    with open(args.src, "r", encoding="latin-1") as fh:
        data = json.load(fh)
    feats = data.get("features", [])
    print(f"  {len(feats):,} features")

    rows = []
    geoms = []
    skipped = 0
    for ft in feats:
        geom_json = ft.get("geometry")
        if not geom_json:
            skipped += 1
            continue
        try:
            geom = shape(geom_json)
        except Exception:
            skipped += 1
            continue
        if geom.is_empty:
            skipped += 1
            continue

        pr = ft.get("properties", {}) or {}
        addr = compose_address(pr)
        city = pr.get("scity") or pr.get("city") or ""
        try:
            lat = float(pr["lat"]) if _nn(pr.get("lat")) else geom.centroid.y
            lon = float(pr["lon"]) if _nn(pr.get("lon")) else geom.centroid.x
        except (TypeError, ValueError):
            lat, lon = geom.centroid.y, geom.centroid.x

        rows.append({
            "parcelnumb": pr.get("parcelnumb") or "",
            "address": addr,
            "city": str(city).title() if city else "",
            "full_address": (f"{addr}, {str(city).title()}, IL" if addr and city
                             else (addr or "")),
            "gisacre": float(pr["gisacre"]) if _nn(pr.get("gisacre")) else None,
            "lat": lat,
            "lon": lon,
        })
        geoms.append(geom)

    gdf = gpd.GeoDataFrame(pd.DataFrame(rows), geometry=geoms, crs=WGS84)
    # Drop degenerate geometries the site-plan engine can't use.
    gdf = gdf[gdf.geometry.notna() & (gdf.geometry.area > 0)].reset_index(drop=True)

    print(f"  kept {len(gdf):,} parcels, skipped {skipped:,}")
    print(f"  with a usable address string: "
          f"{(gdf['address'].str.len() > 0).sum():,}")

    gdf.to_parquet(args.dst)
    size_mb = os.path.getsize(args.dst) / 1e6
    print(f"Wrote {args.dst} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
