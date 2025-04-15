import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_objects_custom(image_path, background_path=None, min_object_size=500):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found. Check the image path.")
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    # 1. Convert to grayscale
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # 2. Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Use adaptive thresholding to segment foreground
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV,
                                   blockSize=11, C=2)
    
    # 4. Apply morphological operations to refine the mask
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 5. Find all contours
    contours, _ = cv2.findContours(morph.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        raise ValueError("No objects found. Adjust preprocessing parameters.")
    
    # 6. Create a mask for detected objects
    mask = np.zeros_like(gray)
    
    # Filter out small contours (noise) and keep only large objects
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_object_size:
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # 7. Extract objects using the mask
    extracted_objects = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
    
    # 8. Prepare new background
    if background_path is not None:
        new_bg = cv2.imread(background_path)
        if new_bg is None:
            raise ValueError("Background image not found.")
        new_bg = cv2.cvtColor(new_bg, cv2.COLOR_BGR2RGB)
        new_bg = cv2.resize(new_bg, (width, height))
    else:
        # Create a plain white background
        new_bg = 255 * np.ones_like(image_rgb)
    
    # 9. Create a 3-channel mask for compositing
    mask_3ch = cv2.merge([mask, mask, mask])
    
    # Remove the object region from the new background
    bg_part = cv2.bitwise_and(new_bg, new_bg, mask=cv2.bitwise_not(mask))
    # Add the extracted objects
    final = cv2.add(bg_part, extracted_objects)
    
    # 10. Display results
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    
    ax[0, 0].imshow(image_rgb)
    ax[0, 0].set_title("Original Image")
    ax[0, 0].axis("off")
    
    ax[0, 1].imshow(mask, cmap="gray")
    ax[0, 1].set_title("Detected Mask")
    ax[0, 1].axis("off")
    
    ax[1, 0].imshow(extracted_objects)
    ax[1, 0].set_title("Extracted Objects")
    ax[1, 0].axis("off")
    
    ax[1, 1].imshow(final)
    ax[1, 1].set_title("Final Composite with New Background")
    ax[1, 1].axis("off")
    
    plt.tight_layout()
    plt.show()

# Example usage
image_path = "car3.jpeg"         # Replace with input image path
background_path = "sky.jpeg"  # Replace with background image path (or set to None)
extract_objects_custom(image_path, background_path)
