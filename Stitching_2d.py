import sys
import time
from pathlib import Path

import cv2
import numpy as np

IMAGE_DIR = Path('C:/Users/edgar/Projects/ODIN/example_quarry_2.0/images/images')
OUTPUT_PATH = Path(sys.argv[1] if len(sys.argv) > 1 else 'orb_mosaic.png')
MAX_IMAGES = int(sys.argv[2]) if len(sys.argv) > 2 else 0   # 0 = all

WORK_LONG_SIDE = 1500   # images are downscaled to this before detection
MAX_PX = 6000           # long side of the output canvas (~300MB of buffers)
WINDOW = 3 #2              # previous frames each image is aligned against
NUM_FEATURES = 7000     # matching is O(n^2), so this sets the runtime
RATIO = 0.80            # Lowe ratio on Hamming distances
MIN_MATCHES = 30
NUM_BANDS = 5


#Downsamples the image to a long side of WORK_LONG_SIDE, then detects ORB keypoints and descriptors.
#Returns the keypoints as an array, the descriptors, and the shape of the downsampled image.
def detect(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read {path.name}")
    factor = WORK_LONG_SIDE / max(image.shape)
    small = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    keypoints, descriptors = cv2.ORB_create(nfeatures=NUM_FEATURES, fastThreshold=7).detectAndCompute(small, None)
    return (np.array([kp.pt for kp in keypoints], np.float32).reshape(-1, 2), descriptors, small.shape)

# Matches descriptors from two images using a brute-force matcher and applies Lowe's ratio test to filter good matches
# the best match has to be significantly better than the second best match. Returns an array of index pairs of matching descriptors.
def match(desc_a, desc_b):
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return np.zeros((0, 2), int)
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_a, desc_b, k=2)
    good = [(p[0].queryIdx, p[0].trainIdx) for p in pairs
            if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
    return np.array(good, int) if good else np.zeros((0, 2), int)

def place(transform, shape):
    h, w = shape
    sw, sh = max(1, round(w * render)), max(1, round(h * render))
    # Exact source->working scale, rather than assuming 1/render, so the
    # rounding above cannot introduce a sub-pixel shift.
    matrix = scale(render) @ recentre @ transform @ np.diag([w / sw, h / sh, 1.0])
    box = (np.array([[0, 0, 1], [sw, 0, 1], [sw, sh, 1], [0, sh, 1]], float) @ matrix.T)[:, :2]
    x0, y0 = np.maximum(np.floor(box.min(0)), 0).astype(int)
    x1, y1 = np.minimum(np.ceil(box.max(0)) + 1, [width, height]).astype(int)
    return matrix, (sw, sh), (x0, y0, x1, y1)


def border_weight(sw, sh):
    border = np.full((sh, sw), 255, np.uint8)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = 0
    weight = cv2.distanceTransform(border, cv2.DIST_L2, 3)
    return (weight / max(weight.max(), 1e-6)).astype(np.float32)

# Fits a rigid transformation (rotation + translation) between two sets of points using Singular Value Decomposition (SVD).
#It takes the centers and figutres out the rotation and translation that best aligns the source points to the destination points
# If there is shearing or scaling its over, it returns a 3x3 matrix
def fit_rigid(src, dst):
    src_centre, dst_centre = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - src_centre).T @ (dst - dst_centre))
    flip = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, flip]) @ u.T
    return np.vstack([np.hstack([rotation,
                                 (dst_centre - rotation @ src_centre).reshape(2, 1)]),
                      [0, 0, 1]])

paths = sorted(IMAGE_DIR.glob('*.JPG'))[:MAX_IMAGES or None]
if not paths:
    raise ValueError(f"No .JPG files")
print(f"Stitching {len(paths)} images in capture order")

t0 = time.perf_counter()
features = [detect(p) for p in paths]
print(f"Detection done in {time.perf_counter() - t0:.1f}s")

t0 = time.perf_counter()
transforms = [np.eye(3)]
breaks = 0
for i in range(1, len(features)):
    src, dst = [], []
    for j in range(max(0, i - WINDOW), i):
        pairs = match(features[i][1], features[j][1])
        if len(pairs) < MIN_MATCHES:
            continue
        there = features[j][0][pairs[:, 1]]
        src.append(features[i][0][pairs[:, 0]])
        dst.append((np.hstack([there, np.ones((len(there), 1), np.float32)])
                    @ transforms[j].T)[:, :2])

    mask = None
    if src:
        src, dst = np.vstack(src).astype(float), np.vstack(dst).astype(float)
        # RANSAC only to get an inlier mask that ORB's ~20% bad matches
        # cannot skew; the inliers are then refitted rigidly.
        _, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                              ransacReprojThreshold=3.0, maxIters=5000)
    if mask is None or mask.sum() < 3:
        print(f"  BREAK at {paths[i].name}: everything after this is a guess")
        breaks += 1
        transforms.append(transforms[-1].copy())
        continue
    keep = mask.ravel().astype(bool)
    transforms.append(fit_rigid(src[keep], dst[keep]))
print(f"Alignment done in {time.perf_counter() - t0:.1f}s, {breaks} break(s)")

corners = np.vstack([
    (np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float) @ t.T)[:, :2]
    for t, (_, _, (h, w)) in zip(transforms, features)])
lo, hi = corners.min(0), corners.max(0)
render = min(1.0, MAX_PX / max(hi - lo))
width, height = np.ceil((hi - lo) * render).astype(int)
scale = lambda k: np.array([[k, 0, 0], [0, k, 0], [0, 0, 1.0]])
recentre = np.array([[1, 0, -lo[0]], [0, 1, -lo[1]], [0, 0, 1]], float)
print(f"Canvas {width}x{height} at render scale {render:.3f}")
canvas = np.zeros((height, width, 3), np.uint8)
best = np.zeros((height, width), np.float32)
owner = np.full((height, width), -1, np.int32)
best = np.zeros((height, width), np.float32)
placements = []

for i, (transform, (_, _, shape)) in enumerate(zip(transforms, features)):
    matrix, (sw, sh), (x0, y0, x1, y1) = place(transform, shape)
    placements.append((matrix, (sw, sh), (x0, y0, x1, y1)))
    if x1 <= x0 or y1 <= y0:
        continue
    to_roi = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], float) @ matrix
    warped = cv2.warpPerspective(border_weight(sw, sh), to_roi, (x1 - x0, y1 - y0),
                                 flags=cv2.INTER_LINEAR)
    region = best[y0:y1, x0:x1]
    wins = warped > region
    region[wins] = warped[wins]
    owner[y0:y1, x0:x1][wins] = i

# Stage 2: warp each frame into its own ROI crop, not a full-canvas copy
t0 = time.perf_counter()
crops, masks, origins = [], [], []
for i, (path, (matrix, (sw, sh), (x0, y0, x1, y1))) in enumerate(zip(paths, placements)):
    if x1 <= x0 or y1 <= y0:
        continue
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    small = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
    to_roi = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], float) @ matrix
    crops.append(cv2.warpPerspective(small, to_roi, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR))
    masks.append(((owner[y0:y1, x0:x1] == i) * 255).astype(np.uint8))
    origins.append((int(x0), int(y0)))
print(f"Warped {len(crops)} frames in {time.perf_counter() - t0:.1f}s")

# Per-frame, per-channel gain solved so overlaps agree
t0 = time.perf_counter()
compensator = cv2.detail.ExposureCompensator_createDefault(
    cv2.detail.ExposureCompensator_CHANNELS)
compensator.feed(origins, crops, masks)
for i in range(len(crops)):
    compensator.apply(i, origins[i], crops[i], masks[i])
print(f"Exposure compensation in {time.perf_counter() - t0:.1f}s")

blender = cv2.detail.MultiBandBlender()
blender.setNumBands(NUM_BANDS)
blender.prepare((0, 0, int(width), int(height)))
for crop, mask, origin in zip(crops, masks, origins):
    blender.feed(crop.astype(np.int16), mask, origin)
blended, _ = blender.blend(None, None)
print(f"Multi-band blend in {time.perf_counter() - t0:.1f}s")

cv2.imwrite(str(OUTPUT_PATH), np.clip(blended, 0, 255).astype(np.uint8))
print(f"Coverage {100.0 * (owner >= 0).mean():.1f}% -> {OUTPUT_PATH}")