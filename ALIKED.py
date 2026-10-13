#ALIKED (A Lighter Keypoint and Descriptor Extraction network)

import os
# Fixes macOS KMP error
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import matplotlib.pyplot as plt
from lightglue import ALIKED
from lightglue.utils import load_image

# Loading the image
imagepath = './Sample_image.JPG'

# Set to None to run on the full resolution image.
RESIZE = 2048

# Cuda, MPS, or CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

# load_image returns an RGB float tensor of shape (3, H, W) with values in [0, 1]
try:
    image = load_image(imagepath)
except FileNotFoundError:
    raise ValueError(f"Could not load image: {imagepath}")

#initializing ALIKED
# max_num_keypoints keeps the N strongest keypoints (-1 = keep everything above the threshold)
extractor = ALIKED(max_num_keypoints=8192, detection_threshold=0.2).eval().to(device)

# Detecting keypoints and computing descriptors
# Keypoints are (x, y) pixel coordinates in the original image
# Descriptors are 128-dim vectors describing the patch around each keypoint
with torch.inference_mode():
    feats = extractor.extract(image.to(device), resize=RESIZE)

# Outputs have a batch dimension of 1, so take [0]
keypoints = feats['keypoints'][0].cpu().numpy()        # (N, 2)
scores = feats['keypoint_scores'][0].cpu().numpy()     # (N,)
descriptors = feats['descriptors'][0].cpu().numpy()    # (N, 128)

print(f"Number of keypoints detected: {len(keypoints)}")
print(f"Shape of descriptors: {descriptors.shape}")

# Plotting the keypoints on the image, coloured by detection score
plt.figure(figsize=(10, 8))
plt.imshow(image.permute(1, 2, 0).numpy())
plt.scatter(keypoints[:, 0], keypoints[:, 1], c=scores, s=2, cmap='viridis')
plt.colorbar(label='Keypoint score')
plt.title(f"ALIKED Keypoints (Total: {len(keypoints)})")
plt.axis('off')

output_path = 'aliked_output.png'
plt.savefig(output_path, bbox_inches='tight')
print(f"Saved result to {output_path}")