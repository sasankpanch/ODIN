#SIFT (Scale-Invariant Feature Transform)

import cv2
import matplotlib.pyplot as plt

# Loading the image
imagepath = './Sample_image.JPG' 
img = cv2.imread(imagepath)
# If there is an error in loading the image, raise an error
if img is None:
    raise ValueError("Could not load image")

#converting to greyscale (SIFT needs greyscale images)
greyscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#initializing SIFT detector
sift = cv2.SIFT_create()

# Telling it to detect keypoints and compute descriptors
# Keypoints are the points of interest in the image
# Descriptors are the feature vectors that describe the local image patches around those keypoints
keypoints, descriptors = sift.detectAndCompute(greyscale_img, None)

print(f"Number of keypoints detected: {len(keypoints)}")
print(f"Shape of descriptors: {descriptors.shape}")

output_img = cv2.drawKeypoints(greyscale_img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
plt.title(f"SIFT Keypoints (Total: {len(keypoints)})")
plt.axis('off')

output_path = 'sift_output.png'
plt.savefig(output_path, bbox_inches='tight')
print(f"Saved result to {output_path}")
