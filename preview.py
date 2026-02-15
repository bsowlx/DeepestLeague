import cv2
import os
import matplotlib.pyplot as plt
from glob import glob

# Path to your generated images
img_paths = sorted(glob("data/synthetics/leagueoflegends-synthetic-dataset/train/images/*.png"))[:9]

plt.figure(figsize=(10, 10))
for i, img_path in enumerate(img_paths):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(3, 3, i + 1)
    plt.imshow(img)
    plt.axis('off')

plt.tight_layout()
plt.savefig("preview.jpg")
print("Saved preview.jpg! Upload this to your README.")