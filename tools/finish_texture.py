# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow", "numpy"]
# ///
"""Finish the baked island textures exported from Blender for the web.

Blender's colour transform turns the baked sea a greyer blue on save, and the bake leaves a dark
rim along the image edge. This shifts every sea-coloured pixel to the page's exact sea colour
(blending the coastline) and paints a thin border of sea, so the terrain meets the page's own
sea plane without a seam.

Run:  uv run tools/finish_texture.py assets/island/island_2k.jpg assets/island/island_4k.jpg
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

PAGE_SEA = np.array([0xB9, 0xD8, 0xE4], np.float32)  # --sea in index.html
TOLERANCE = 6     # colour distance treated as pure sea
FEATHER = 30      # distance over which coastline pixels blend toward the page sea
BORDER_PX = 4


def finish(path: Path) -> None:
    rgb = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)
    height, width, _ = rgb.shape
    open_ocean = rgb[height // 2:, : width // 5].reshape(-1, 3)  # lower left of the frame is open Pacific
    baked_sea = np.median(open_ocean, axis=0)

    distance = np.linalg.norm(rgb - baked_sea, axis=2)
    weight = np.clip(1 - (distance - TOLERANCE) / FEATHER, 0, 1)[..., None]
    out = rgb + (PAGE_SEA - baked_sea) * weight

    b = BORDER_PX
    out[:b], out[-b:], out[:, :b], out[:, -b:] = PAGE_SEA, PAGE_SEA, PAGE_SEA, PAGE_SEA
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(path, quality=86, optimize=True, progressive=True)
    print(f"{path.name}: baked sea {baked_sea.astype(int)} -> page sea, {int((weight > 0).sum())} px moved")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        finish(Path(arg))
