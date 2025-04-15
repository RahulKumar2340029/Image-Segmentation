import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Parameters to Tune ---
# Define the range for the BACKGROUND color (white/light gray) in HSV
# White/Gray usually has LOW Saturation and HIGH Value. Hue can be broad.
# You WILL likely need to ADJUST these based on your specific image's lighting/shadows.
lower_background_hsv = np.array([0, 0, 150])     # Lower bound (Min Hue, Min Sat, Min Brightness)
upper_background_hsv = np.array([179, 70, 255])  # Upper bound (Max Hue, Max Sat, Max Brightness)
# Note: Hue wraps around (0/179 are reds), Sat 0-~70 covers grays/whites, Val >~150 covers bright areas

# Morphological Kernel Size (May need adjustment)
kernel_size_closing = 10 # Closing might help fill shadow gaps in background
kernel_size_opening = 5  # Opening helps remove small non-background specks
# --- End Parameters ---

# --- File Paths ---
input_image_path = 'lion.jpg'  # Use the same image you showed
background_image_path = 'sky.jpeg' # Change to your background file
# --- End File Paths ---

# 1. Load Images
img = cv2.imread(input_image_path)
background_img = cv2.imread(background_image_path)

if img is None:
    print(f"Error: Could not load input image at {input_image_path}")
    exit()
if background_img is None:
    print(f"Error: Could not load background image at {background_image_path}")
    exit()

# Resize background to match input image size
img_h, img_w = img.shape[:2]
background_img = cv2.resize(background_img, (img_w, img_h))

# 2. Preprocessing - Optional Blur
blurred_img = cv2.GaussianBlur(img, (5, 5), 0)

# 3. Convert to HSV Color Space
hsv_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)

# 4. Create Mask for the BACKGROUND using Color Thresholding
mask_background = cv2.inRange(hsv_img, lower_background_hsv, upper_background_hsv)

# 5. Refine BACKGROUND Mask using Morphological Operations
# Closing first can help bridge gaps in the background (e.g., across shadows)
kernel_close = np.ones((kernel_size_closing, kernel_size_closing), np.uint8)
mask_background_closed = cv2.morphologyEx(mask_background, cv2.MORPH_CLOSE, kernel_close, iterations=1)

# Opening removes small noise elements (pixels falsely identified as background)
kernel_open = np.ones((kernel_size_opening, kernel_size_opening), np.uint8)
mask_background_opened = cv2.morphologyEx(mask_background_closed, cv2.MORPH_OPEN, kernel_open, iterations=1)

# This is the refined mask identifying the BACKGROUND
refined_background_mask = mask_background
# refined_background_mask = mask_background_opened

# 6. *** INVERT the background mask to get the FOREGROUND mask ***
final_mask = cv2.bitwise_not(refined_background_mask)

# Create an inverse mask for the NEW background (this will be the refined_background_mask)
inverse_mask_for_new_bg = refined_background_mask # Pixels that ARE background

# 7. Apply Masks to Extract Foreground and Background Parts
# Extract foreground from the original image using the inverted mask
foreground = cv2.bitwise_and(img, img, mask=final_mask)

# Extract the relevant part of the new background image using the background mask
new_background_part = cv2.bitwise_and(background_img, background_img, mask=inverse_mask_for_new_bg)

# 8. Combine Foreground and New Background
final_result = cv2.add(foreground, new_background_part)

# 9. Display Results using Matplotlib
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(mask_background, cmap='gray') # Show initial background mask
plt.title('Initial Background Mask')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(refined_background_mask, cmap='gray') # Show refined background mask
plt.title('Refined Background Mask')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(final_mask, cmap='gray') # Show the INVERTED mask (Foreground)
plt.title('Final Foreground Mask (Inverted)')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB))
plt.title('Extracted Foreground')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB))
plt.title('Final Result (New Background)')
plt.axis('off')

plt.tight_layout()
plt.show()

# Optionally save the result
# cv2.imwrite('foreground_extraction_result_v2.jpg', final_result)
# cv2.imwrite('foreground_mask_v2.jpg', final_mask)