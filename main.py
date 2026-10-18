import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from rembg import remove
from skimage.filters import sobel
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import RMSprop

print("=== Starting End-to-End Image Processing & ML Pipeline ===")

# ==========================================
# STEP 1: Directory Setup & Synthetic Image Fallback
# ==========================================
os.makedirs('original', exist_ok=True)
os.makedirs('masked', exist_ok=True)
os.makedirs('output_results', exist_ok=True)

image_path = 'input_image.jpg'

# If no input image exists, create a dummy test image so the script runs smoothly
if not os.path.exists(image_path):
    print(f"[{image_path}] not found. Generating a sample test image...")
    dummy_img = np.random.randint(60, 220, (400, 400, 3), dtype=np.uint8)
    cv2.rectangle(dummy_img, (100, 100), (300, 300), (255, 255, 255), -1)
    cv2.imwrite(image_path, dummy_img)


# ==========================================
# STEP 2: Loading & Resizing Image
# ==========================================
print("\n--- Step 2: Loading and Resizing Image ---")
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Could not load image from {image_path}")

img = cv2.resize(img, (400, 400))
height, width = img.shape[:2]
print(f"Image loaded successfully with dimensions: {width}x{height}")


# ==========================================
# STEP 3: Brightness & Contrast Adjustment
# ==========================================
print("\n--- Step 3: Adjusting Brightness & Contrast ---")
brightness = 2
contrast = 1.2
enhanced_img = cv2.addWeighted(img, contrast, np.zeros(img.shape, img.dtype), 0, brightness)
cv2.imwrite('output_results/enhanced_image.jpg', enhanced_img)
print("Enhanced image saved to 'output_results/enhanced_image.jpg'")


# ==========================================
# STEP 4: Grayscale Conversion & Edge Detection
# ==========================================
print("\n--- Step 4: Grayscale Conversion & Edge Detection ---")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Canny Edge Detection
canny_edges = cv2.Canny(gray_img, 40, 250)

# Sobel Edge Detection
sobel_edges = sobel(gray_img)
sobel_uint8 = (sobel_edges * 255).astype(np.uint8)
print("Edge detection completed (Canny & Sobel).")


# ==========================================
# STEP 5: Thresholding & Contour Extraction
# ==========================================
print("\n--- Step 5: Thresholding & Contour Analysis ---")
# Otsu's Thresholding
_, binary_thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Find Contours
contours, hierarchy = cv2.findContours(binary_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
print(f"Total contours found: {len(contours)}")

# Sort contours by area descending
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)


# ==========================================
# STEP 6: Region of Interest (ROI) Extraction
# ==========================================
print("\n--- Step 6: Extracting Regions of Interest (ROIs) ---")
roi_count = 0
for cnt in sorted_contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 15 and h > 15:  # Filter out very small noise
        roi = img[y:y + h, x:x + w]
        roi_count += 1
print(f"Successfully extracted {roi_count} valid ROI bounding boxes.")


# ==========================================
# STEP 7: Advanced Segmentation (GrabCut & rembg)
# ==========================================
print("\n--- Step 7: Running GrabCut & Background Removal ---")
# GrabCut Foreground Extraction
mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

rect = (50, 50, 300, 300) # Bounding box within 400x400
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
grabcut_result = img * mask2[:, :, np.newaxis]

# AI Background Removal via rembg
pil_img = Image.open(image_path)
pil_img.save(os.path.join('original', 'source.jpg'), format='jpeg')

with open(os.path.join('masked', 'output_removed.png'), 'wb') as f:
    input_bytes = open(os.path.join('original', 'source.jpg'), 'rb').read()
    subject_bytes = remove(input_bytes)
    f.write(subject_bytes)
print("Background removal and GrabCut segmentation completed.")


# ==========================================
# STEP 8: Deep Learning CNN Pipeline Setup
# ==========================================
print("\n--- Step 8: Initializing Deep Learning CNN Model ---")
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(200, 200, 3)),
    MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'),
    
    Conv2D(64, kernel_size=(3, 3), padding='valid', activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'),
    
    Conv2D(128, kernel_size=(3, 3), padding='valid', activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.001), metrics=['accuracy'])
print("CNN Model compiled successfully!")

print("\n=== Pipeline Execution Completed Successfully! ===")