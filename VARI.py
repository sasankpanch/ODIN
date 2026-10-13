#VARI (Visible Atmospherically Resistant Index) vegetation index
# VARI = (G - R) / (G + R - B) needs only the RGB channels, so it works on
# plain drone/aerial JPGs with no near-infrared band. Higher values mean
# greener pixels. It is a rough visible-band proxy, not a calibrated measure
# of plant health like NDVI, but it is enough to separate foliage from
# bare ground, roads, roofs and sky.zmtell

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFile, ImageOps

# Lets Pillow decode a JPEG whose final bytes are missing (common after an
# interrupted copy). The bottom edge may look slightly off as a result.
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_PATH = Path('./ER-02-21260_0129_0098.JPG')
OUTPUT_PATH = Path('./vari_output.png')
MASK_PATH = Path('./vari_mask.png')

# Pixels with VARI above this are labelled foliage. 0 is a neutral starting
# point; raise it to be stricter (fewer false positives from green-tinted
# ground), lower it to catch dry or pale vegetation.
THRESHOLD = 0.05

# Pixels darker than this (0 to 1 scale) are ignored as unreliable
MIN_BRIGHTNESS = 0.05

# Downscale for display and speed; set to None to use the full resolution.
MAX_SIDE = 2048

# ImageOps.exif_transpose applies the EXIF rotation so the output matches
# how the photo looks when viewed normally.
if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Could not find {IMAGE_PATH}")
img = ImageOps.exif_transpose(Image.open(IMAGE_PATH)).convert('RGB')
if MAX_SIDE is not None:
    img.thumbnail((MAX_SIDE, MAX_SIDE))

rgb = np.asarray(img, dtype=np.float64) / 255.0
r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

# Very dark pixels make the denominator tiny, so VARI blows up to meaningless
# values. Those pixels are excluded, and the rest are clipped to [-1, 1].
brightness = rgb.mean(axis=2)
denom = g + r - b
valid = (brightness > MIN_BRIGHTNESS) & (np.abs(denom) > 1e-3)
vari = np.zeros_like(brightness)
vari[valid] = np.clip((g - r)[valid] / denom[valid], -1.0, 1.0)
foliage = valid & (vari > THRESHOLD)

coverage = 100.0 * foliage.mean()
print(f"Image size: {img.width}x{img.height}")
print(f"VARI range: {vari.min():.3f} to {vari.max():.3f}, median {np.median(vari):.3f}")
print(f"Foliage threshold: {THRESHOLD}")
print(f"Foliage coverage: {coverage:.1f}% of pixels")

# Binary mask saved on its own, so it can be reused by the matching pipelines
Image.fromarray((foliage * 255).astype(np.uint8)).save(MASK_PATH)
print(f"Saved mask to {MASK_PATH}")

# Three panels: original, VARI map, and the foliage mask overlaid in red
overlay = rgb.copy()
overlay[foliage] = 0.5 * overlay[foliage] + 0.5 * np.array([1.0, 0.0, 0.0])

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(rgb)
axes[0].set_title('Ground Truth')
im = axes[1].imshow(vari, cmap='RdYlGn', vmin=-0.3, vmax=0.3)
axes[1].set_title(f'VARI (green = more)')
fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
axes[2].imshow(overlay)
axes[2].set_title(f'Foliage mask (VARI > {THRESHOLD}): {coverage:.1f}%')
for ax in axes:
    ax.axis('off')

fig.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=150)
print(f"Saved result to {OUTPUT_PATH}")
