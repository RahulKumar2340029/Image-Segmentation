import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import io
import tempfile
import os

def process_image(img, background_img, lower_background_hsv, upper_background_hsv, 
                  kernel_size_closing, kernel_size_opening):
    """
    Process an image to replace its background
    """
    # Resize background to match input image size
    img_h, img_w = img.shape[:2]
    background_img = cv2.resize(background_img, (img_w, img_h))

    # Preprocessing - Optional Blur
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)

    # Convert to HSV Color Space
    hsv_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)

    # Create Mask for the BACKGROUND using Color Thresholding
    mask_background = cv2.inRange(hsv_img, lower_background_hsv, upper_background_hsv)

    # Refine BACKGROUND Mask using Morphological Operations
    # Closing helps bridge gaps in the background (e.g., across shadows)
    kernel_close = np.ones((kernel_size_closing, kernel_size_closing), np.uint8)
    mask_background_closed = cv2.morphologyEx(mask_background, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # Opening removes small noise elements
    kernel_open = np.ones((kernel_size_opening, kernel_size_opening), np.uint8)
    mask_background_opened = cv2.morphologyEx(mask_background_closed, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # Use refined mask as needed
    if st.session_state.use_morphology:
        refined_background_mask = mask_background_opened
    else:
        refined_background_mask = mask_background

    # INVERT the background mask to get the FOREGROUND mask
    final_mask = cv2.bitwise_not(refined_background_mask)

    # Inverse mask for the NEW background
    inverse_mask_for_new_bg = refined_background_mask

    # Apply Masks to Extract Foreground and Background Parts
    foreground = cv2.bitwise_and(img, img, mask=final_mask)
    new_background_part = cv2.bitwise_and(background_img, background_img, mask=inverse_mask_for_new_bg)

    # Combine Foreground and New Background
    final_result = cv2.add(foreground, new_background_part)

    return {
        'original': img,
        'background_mask': mask_background,
        'refined_mask': refined_background_mask,
        'foreground_mask': final_mask,
        'foreground': foreground,
        'final_result': final_result
    }

def main():
    st.set_page_config(page_title="Background Replacement App", layout="wide")
    st.title("Background Replacement Tool")
    st.write("Upload an image to extract the foreground and replace the background")

    # Initialize session state variables if they don't exist
    if 'lower_h' not in st.session_state:
        st.session_state.lower_h = 0
        st.session_state.lower_s = 0
        st.session_state.lower_v = 150
        st.session_state.upper_h = 179
        st.session_state.upper_s = 70
        st.session_state.upper_v = 255
        st.session_state.kernel_closing = 10
        st.session_state.kernel_opening = 5
        st.session_state.use_morphology = True

    # Create two columns for the upload widgets
    col1, col2 = st.columns(2)

    with col1:
        # Input image upload
        input_image = st.file_uploader("Upload foreground image", type=["jpg", "jpeg", "png"])
    
    with col2:
        # Background image upload
        background_image = st.file_uploader("Upload background image", type=["jpg", "jpeg", "png"])
    
    # Create sidebar for parameter tuning
    st.sidebar.header("Parameter Tuning")
    
    # HSV Range for Background Detection
    st.sidebar.subheader("Background Color Range (HSV)")
    
    # Create columns for the HSV sliders
    col_h1, col_h2 = st.sidebar.columns(2)
    with col_h1:
        st.session_state.lower_h = st.slider("Min Hue", 0, 179, st.session_state.lower_h)
    with col_h2:
        st.session_state.upper_h = st.slider("Max Hue", 0, 179, st.session_state.upper_h)
    
    col_s1, col_s2 = st.sidebar.columns(2)
    with col_s1:
        st.session_state.lower_s = st.slider("Min Saturation", 0, 255, st.session_state.lower_s)
    with col_s2:
        st.session_state.upper_s = st.slider("Max Saturation", 0, 255, st.session_state.upper_s)
    
    col_v1, col_v2 = st.sidebar.columns(2)
    with col_v1:
        st.session_state.lower_v = st.slider("Min Value", 0, 255, st.session_state.lower_v)
    with col_v2:
        st.session_state.upper_v = st.slider("Max Value", 0, 255, st.session_state.upper_v)
    
    # Morphological Operations
    st.sidebar.subheader("Morphological Operations")
    st.session_state.use_morphology = st.sidebar.checkbox("Apply Morphological Operations", st.session_state.use_morphology)
    
    if st.session_state.use_morphology:
        st.session_state.kernel_closing = st.sidebar.slider("Closing Kernel Size", 1, 30, st.session_state.kernel_closing)
        st.session_state.kernel_opening = st.sidebar.slider("Opening Kernel Size", 1, 30, st.session_state.kernel_opening)
    
    # Process images if both are uploaded
    if input_image is not None and background_image is not None:
        try:
            # Save uploaded files to temporary location first
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_input:
                tmp_input.write(input_image.getvalue())
                input_path = tmp_input.name
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_bg:
                tmp_bg.write(background_image.getvalue())
                bg_path = tmp_bg.name
            
            # Read images with OpenCV directly
            input_img = cv2.imread(input_path)
            background_img = cv2.imread(bg_path)
            
            # Delete temporary files
            os.unlink(input_path)
            os.unlink(bg_path)
            
            if input_img is None or background_img is None:
                st.error("Failed to process one or both images. Please try different image files.")
                return
            
            # Set HSV parameters from sliders
            lower_hsv = np.array([st.session_state.lower_h, st.session_state.lower_s, st.session_state.lower_v])
            upper_hsv = np.array([st.session_state.upper_h, st.session_state.upper_s, st.session_state.upper_v])
            
            # Process image with current parameters
            results = process_image(
                input_img, 
                background_img, 
                lower_hsv, 
                upper_hsv, 
                st.session_state.kernel_closing, 
                st.session_state.kernel_opening
            )
            
            # Display results in a grid
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(cv2.cvtColor(results['original'], cv2.COLOR_BGR2RGB))
                
                st.subheader("Background Mask")
                st.image(results['background_mask'], use_column_width=True)
                
                st.subheader("Foreground Mask (Inverted)")
                st.image(results['foreground_mask'], use_column_width=True)
            
            with col2:
                st.subheader("Background Image")
                st.image(cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB))
                
                st.subheader("Extracted Foreground")
                st.image(cv2.cvtColor(results['foreground'], cv2.COLOR_BGR2RGB))
                
                st.subheader("Final Result")
                final_result_rgb = cv2.cvtColor(results['final_result'], cv2.COLOR_BGR2RGB)
                st.image(final_result_rgb, use_column_width=True)
                
                # Add download button for the final result
                result_pil = Image.fromarray(final_result_rgb)
                buf = io.BytesIO()
                result_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="Download Result",
                    data=byte_im,
                    file_name="background_replacement_result.png",
                    mime="image/png"
                )
                
        except Exception as e:
            st.error(f"An error occurred while processing the images: {str(e)}")
            st.error("Please try different image files or check if the images are valid.")
    else:
        st.info("Please upload both a foreground image and a background image to see results.")
        
        # Display demonstration images
        st.subheader("Example of what this app can do:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image("lion.jpg", caption="Original Image Example")
        with col2:
            st.image("sky.jpeg", caption="Background Image Example")
        with col3:
            st.image("result.png", caption="Result Example")

if __name__ == "__main__":
    main()