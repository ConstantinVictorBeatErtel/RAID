#!/usr/bin/env python3
"""Recolor white-background report charts onto the site's dark background.

White background -> #0d1117, dark text/axes/outlines -> light, colored
bars/markers (blues, green, amber) preserved. Anti-aliasing stays smooth
because grayscale pixels are remapped by luminance.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

BG = np.array([13, 17, 23], dtype=float)        # site background
LIGHT = np.array([230, 237, 243], dtype=float)   # site text
SAT_THRESHOLD = 0.16                              # above => treat as colored

REPO = Path(__file__).resolve().parents[1]
FIGURES = REPO / "assets" / "figures"

JOBS = [
    (
        "/Users/ConstiX/.cursor/projects/Users-ConstiX-Downloads-Proteus/assets/image-357069f6-304d-4fff-a108-257af031b6f6.png",
        FIGURES / "val_mse_comparison.png",
    ),
    (
        "/Users/ConstiX/.cursor/projects/Users-ConstiX-Downloads-Proteus/assets/image-68fe2afb-11bd-4cad-9d68-c9f6fabfd924.png",
        FIGURES / "grpo_reward_curve.png",
    ),
]


def recolor(src: str, dst: Path) -> None:
    img = Image.open(src).convert("RGB")
    arr = np.asarray(img, dtype=float)
    mx = arr.max(axis=-1)
    mn = arr.min(axis=-1)
    sat = np.divide(mx - mn, mx, out=np.zeros_like(mx), where=mx > 0)
    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]

    colored = sat > SAT_THRESHOLD

    # Grayscale pixels: invert luminance onto dark theme.
    # white (lum 255) -> BG, black (lum 0) -> LIGHT.
    t = ((255.0 - lum) / 255.0)[..., None]
    gray = BG * (1.0 - t) + LIGHT * t

    out = np.where(colored[..., None], arr, gray)
    out = np.clip(out, 0, 255).astype(np.uint8)
    Image.fromarray(out).save(dst)
    print(f"Saved: {dst}")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for src, dst in JOBS:
        recolor(src, dst)


if __name__ == "__main__":
    main()
