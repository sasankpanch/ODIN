#COLMAP baseline (classical SIFT extraction + matching + incremental SfM)
# This is the "Track A" comparison point for the ALIKED + LightGlue pipeline
# in colmap_aliked_lightglue.py: same images, same camera model, same
# windowed-pair matching strategy, but using COLMAP's built-in SIFT front end.

import json
import shutil
import time
from pathlib import Path

import pycolmap

# Moved to the external SSD: the internal Desktop volume was the suspected
# cause of the earlier stalls/slow throughput during feature extraction.
SSD_ROOT = Path('/Volumes/500GB SSD/Mosaic')
IMAGE_DIR = SSD_ROOT / 'images'
DATABASE_PATH = SSD_ROOT / 'colmap_baseline.db'
OUTPUT_DIR = SSD_ROOT / 'colmap_baseline_sparse'

# How many neighboring frames (in filename order) each image is matched
# against. All 347 images share one camera, captured sequentially, so
# exhaustive all-pairs matching would be wasteful.
SEQUENTIAL_OVERLAP = 10

# Start clean: COLMAP's database/mapping calls assume they're populating
# fresh state, so a leftover db or sparse model from a previous run would
# either error out or silently mix with new data.
if DATABASE_PATH.exists():
    DATABASE_PATH.unlink()
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True)

timings = {}

# All 347 images come from the same camera, so a single shared set of
# intrinsics (COLMAP's default SIMPLE_RADIAL model) keeps the comparison
# against the ALIKED+LightGlue track fair.
t0 = time.perf_counter()
pycolmap.extract_features(
    database_path=DATABASE_PATH,
    image_path=IMAGE_DIR,
    camera_mode=pycolmap.CameraMode.SINGLE,
    # Default num_threads=-1 uses all 10 cores at once. Each thread runs CPU
    # SIFT at full resolution with 2x upsampling (COLMAP's default
    # first_octave=-1) on these 5472x3648 images, which is memory-heavy per
    # thread (COLMAP prints a warning about exactly this). With 10 threads
    # in parallel on this 16GB Mac, that reached 18.5GB and was killed by
    # the OS's memory-pressure killer (confirmed via `log show`: "kernel:
    # memorystatus: killing largest compressed process Python [9252] 18535
    # MB"). Capping threads keeps memory bounded; feature results are
    # identical, just extracted less in parallel.
    extraction_options=pycolmap.FeatureExtractionOptions(num_threads=3),
)
timings['feature_extraction_sec'] = time.perf_counter() - t0
print(f"Feature extraction done in {timings['feature_extraction_sec']:.1f}s")

t0 = time.perf_counter()
pycolmap.match_sequential(
    database_path=DATABASE_PATH,
    pairing_options=pycolmap.SequentialPairingOptions(overlap=SEQUENTIAL_OVERLAP),
)
timings['matching_sec'] = time.perf_counter() - t0
print(f"Sequential matching done in {timings['matching_sec']:.1f}s")

t0 = time.perf_counter()
reconstructions = pycolmap.incremental_mapping(
    database_path=DATABASE_PATH,
    image_path=IMAGE_DIR,
    output_path=OUTPUT_DIR,
)
timings['mapping_sec'] = time.perf_counter() - t0
print(f"Incremental mapping done in {timings['mapping_sec']:.1f}s")

if not reconstructions:
    raise RuntimeError("Incremental mapping failed to register any images")

# incremental_mapping can produce multiple disjoint models if the scene
# doesn't fully connect; the largest one is the one worth comparing.
best_id, best_rec = max(reconstructions.items(), key=lambda kv: kv[1].num_reg_images())

metrics = {
    'num_models': len(reconstructions),
    'num_reg_images': best_rec.num_reg_images(),
    'num_total_images': len(list(IMAGE_DIR.glob('*.JPG'))),
    'num_points3D': best_rec.num_points3D(),
    'mean_track_length': best_rec.compute_mean_track_length(),
    'mean_reprojection_error': best_rec.compute_mean_reprojection_error(),
    **timings,
}

print("\n=== COLMAP baseline results ===")
for key, value in metrics.items():
    print(f"{key}: {value}")

with open(OUTPUT_DIR / 'metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"\nSaved metrics to {OUTPUT_DIR / 'metrics.json'}")
print(f"Sparse model(s) written to {OUTPUT_DIR}/<model_id>/")
