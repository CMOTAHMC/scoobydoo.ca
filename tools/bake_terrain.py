# /// script
# requires-python = ">=3.11"
# dependencies = ["rasterio>=1.4", "numpy"]
# ///
"""Bake a Vancouver Island heightmap from NRCan MRDEM-30 (Open Government Licence - Canada).

Reads the cloud-optimised DEM over HTTP, reprojects to BC Albers (EPSG:3005), downsamples,
and writes a 16-bit PNG heightmap plus a JSON sidecar with bounds and elevation scale.

Run:  uv run tools/bake_terrain.py
"""

import json
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

DEM_URL = "/vsicurl/https://canelevation-dem.s3.ca-central-1.amazonaws.com/mrdem-30/mrdem-30-dtm.vrt"
LONLAT_BOUNDS = (-128.6, 48.2, -123.0, 51.0)  # west, south, east, north - island plus Strait of Georgia
TARGET_CRS = CRS.from_epsg(3005)
WIDTH_PX = 2048
SEA_LEVEL_CUTOFF_M = 1.0
OUT_DIR = Path(__file__).resolve().parent / "terrain"  # Blender input only; not served by the site

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.vrt",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_CACHEMAX": 512,
}


def main() -> None:
    west, south, east, north = transform_bounds("EPSG:4326", TARGET_CRS, *LONLAT_BOUNDS)
    height_px = round(WIDTH_PX * (north - south) / (east - west))
    transform = from_bounds(west, south, east, north, WIDTH_PX, height_px)

    with rasterio.Env(**GDAL_ENV), rasterio.open(DEM_URL) as src:
        with WarpedVRT(
            src,
            crs=TARGET_CRS,
            transform=transform,
            width=WIDTH_PX,
            height=height_px,
            resampling=Resampling.bilinear,
        ) as vrt:
            dem = vrt.read(1, masked=True)

    elevation = dem.filled(0).astype(np.float32)
    elevation[elevation < SEA_LEVEL_CUTOFF_M] = 0  # MRDEM stores sea surface as ~0.1-0.2 m, not no-data
    max_m = float(elevation.max())
    heightmap = np.round(elevation / max_m * 65535).astype(np.uint16)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / "island_height.png"
    with rasterio.open(png_path, "w", driver="PNG", width=WIDTH_PX, height=height_px, count=1, dtype="uint16") as dst:
        dst.write(heightmap, 1)
    png_path.with_suffix(".png.aux.xml").unlink(missing_ok=True)

    meta = {
        "source": "NRCan MRDEM-30 DTM, Open Government Licence - Canada",
        "crs": "EPSG:3005",
        "bounds_lonlat": LONLAT_BOUNDS,
        "bounds_crs": [west, south, east, north],
        "width_px": WIDTH_PX,
        "height_px": height_px,
        "metres_per_px": (east - west) / WIDTH_PX,
        "max_elevation_m": max_m,
        "land_fraction": float((elevation > 0).mean()),
    }
    (OUT_DIR / "island_height.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
