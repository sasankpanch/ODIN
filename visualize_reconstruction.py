#Visualizes a COLMAP sparse reconstruction as a top-down-ish point cloud
# projection + camera trajectory, so pipeline performance can be eyeballed
# without needing COLMAP's GUI (which isn't installed) or a full dense
# mesh/orthomosaic (which needs far more images registered to be meaningful).
#
# Reusable: point SPARSE_DIR at any pipeline's output (colmap_baseline_sparse,
# and later the ALIKED+LightGlue one) to compare them with the same view.

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pycolmap

SPARSE_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else '/Volumes/500GB SSD/Mosaic/colmap_baseline_sparse')
OUTPUT_PATH = Path(sys.argv[2] if len(sys.argv) > 2 else f'{SPARSE_DIR.name}_pointcloud.png')

# incremental_mapping can produce multiple disjoint sub-models (one per
# connected component it managed to register); only the largest is worth
# looking at.
model_dirs = [d for d in SPARSE_DIR.iterdir() if d.is_dir()]
if not model_dirs:
    raise ValueError(f"No sub-model directories found under {SPARSE_DIR}")

reconstructions = [pycolmap.Reconstruction(str(d)) for d in model_dirs]
rec = max(reconstructions, key=lambda r: r.num_reg_images())

points = np.array([p.xyz for p in rec.points3D.values()])
colors = np.array([p.color for p in rec.points3D.values()]) / 255.0
cameras = np.array([im.projection_center() for im in rec.images.values()])

print(f"Visualizing model with {rec.num_reg_images()} registered images, {rec.num_points3D()} points")

# World axes out of incremental SfM are arbitrary (no gravity/GPS alignment
# applied yet), so rather than guess which axis is "up" and mislabel a plot
# as a top-down orthomosaic, show all three axis-pairs and let the shape of
# the point cloud tell you which one actually looks top-down.
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
pairs = [(0, 1, 'X', 'Y'), (0, 2, 'X', 'Z'), (1, 2, 'Y', 'Z')]

for ax, (i, j, xi, xj) in zip(axes, pairs):
    ax.scatter(points[:, i], points[:, j], c=colors, s=1, alpha=0.6)
    ax.plot(cameras[:, i], cameras[:, j], 'r.-', markersize=4, linewidth=0.8, label='camera path')
    ax.set_xlabel(xi)
    ax.set_ylabel(xj)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)

fig.suptitle(
    f"{SPARSE_DIR.name}: {rec.num_reg_images()} images registered, "
    f"{rec.num_points3D()} sparse points (not a dense orthomosaic)"
)
fig.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=150)
print(f"Saved to {OUTPUT_PATH}")
