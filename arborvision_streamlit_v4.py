import streamlit as st
import os
import numpy as np
import cv2
from PIL import Image
import io
import logging
import torch
from typing import Tuple, Optional
import time

# Import functions from arbresV4.py
from arbresv4 import (
    load_image, 
    segment_tree_multi_approach, 
    detect_tree_extremes, 
    estimate_height_and_dbh, 
    render_overlay,
    create_vegetation_mask_by_color,
    try_maskrcnn_segmentation,
    create_fallback_mask,
    refine_tree_mask
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("arborvision_streamlit")

# Set page configuration
st.set_page_config(
    page_title="ArborVision - Tree Analysis", 
    page_icon="🌲",
    layout="wide"
)

def main():
    st.title("🌲 ArborVision - Tree Analysis")
    
    # Sidebar
    st.sidebar.header("Settings")
    
    # Maximum image size
    max_size = st.sidebar.slider("Maximum image size (px)", 400, 2000, 1600, 100)
    
    # Optional manual pixels per meter
    use_manual_ppm = st.sidebar.checkbox("Use manual pixels per meter", False)
    pixels_per_meter = None
    if use_manual_ppm:
        pixels_per_meter = st.sidebar.number_input("Pixels per meter", 1.0, 5000.0, 1000.0, 10.0)
    
    # Select segmentation method
    segmentation_method = st.sidebar.selectbox(
        "Segmentation method", 
        ["Auto (best available)", "Color-based", "Mask R-CNN (if available)", "Shape-based"]
    )
    
    # Upload image
    st.header("Upload an image of a tree")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    # Camera input option
    use_camera = st.checkbox("Or use camera", False)
    camera_image = None
    if use_camera:
        camera_image = st.camera_input("Take a photo")
    
    # Process the image
    if uploaded_file is not None or camera_image is not None:
        col1, col2 = st.columns(2)
        
        with st.spinner("Processing image..."):
            try:
                # Get image data
                if uploaded_file is not None:
                    image_data = uploaded_file.read()
                    image_source = "upload"
                else:
                    image_data = camera_image.read()
                    image_source = "camera"
                
                # Create a temporary file to process with OpenCV
                temp_file = f"temp_{int(time.time())}.jpg"
                with open(temp_file, "wb") as f:
                    f.write(image_data)
                
                # Load and process image
                img_bgr, img_rgb, H, W = load_image(temp_file, max_side=max_size)
                
                # Display original image
                with col1:
                    st.subheader("Original Image")
                    st.image(img_rgb, channels="RGB", use_column_width=True)
                
                # Perform segmentation based on selected method
                if segmentation_method == "Color-based":
                    tree_mask = create_vegetation_mask_by_color(img_rgb)
                    if tree_mask is None:
                        st.error("No tree detected with color-based segmentation.")
                        return
                    tree_mask = refine_tree_mask(tree_mask, img_rgb)
                elif segmentation_method == "Mask R-CNN (if available)":
                    if torch.cuda.is_available():
                        device = "cuda"
                    else:
                        device = "cpu"
                    tree_mask = try_maskrcnn_segmentation(img_rgb, device=device)
                    if tree_mask is None:
                        st.error("No tree detected with Mask R-CNN segmentation.")
                        return
                    tree_mask = refine_tree_mask(tree_mask, img_rgb)
                elif segmentation_method == "Shape-based":
                    tree_mask = create_fallback_mask(img_rgb)
                    if tree_mask is None:
                        st.error("No tree detected with shape-based segmentation.")
                        return
                    tree_mask = refine_tree_mask(tree_mask, img_rgb)
                else:  # Auto (best available)
                    tree_mask = segment_tree_multi_approach(img_rgb, device="cpu")
                
                # Display mask
                with col2:
                    st.subheader("Tree Segmentation")
                    mask_display = np.zeros_like(img_rgb)
                    mask_display[tree_mask > 0] = [0, 255, 0]  # Green mask
                    alpha = 0.5
                    overlay = cv2.addWeighted(img_rgb, 1-alpha, mask_display, alpha, 0)
                    st.image(overlay, channels="RGB", use_column_width=True)
                
                # Detect tree extremes
                (base_x, base_y), (top_x, top_y) = detect_tree_extremes(tree_mask)
                
                # Calculate pixel height
                pixel_height = abs(top_y - base_y)
                
                # Estimate height
                height_m, height_ci, size_label, dbh_m = estimate_height_and_dbh(
                    pixel_height, manual_ppm=pixels_per_meter
                )
                
                # Render result
                result_image = render_overlay(
                    img_bgr, (base_x, base_y), (top_x, top_y),
                    height_m, height_ci, size_label, dbh_m
                )
                
                # Display results
                st.header("Analysis Results")
                col_result, col_metrics = st.columns(2)
                
                with col_result:
                    st.subheader("Detected Tree")
                    st.image(result_image, channels="BGR", use_column_width=True)
                
                with col_metrics:
                    st.subheader("Tree Measurements")
                    st.metric("Tree Height", f"{height_m:.2f} m", f"±{height_ci:.2f} m")
                    st.metric("Tree Size Classification", size_label)
                    st.metric("Estimated DBH (Diameter at Breast Height)", f"{dbh_m:.2f} m")
                    st.metric("Pixel Height", f"{pixel_height} px")
                    if pixels_per_meter:
                        st.metric("Manual Scale", f"{pixels_per_meter:.1f} px/m")
                
                # Download options
                st.header("Download Results")
                
                # Convert result to bytes for download
                is_success, buffer = cv2.imencode(".jpg", result_image)
                io_buf = io.BytesIO(buffer)
                
                # Download buttons
                col_download1, col_download2 = st.columns(2)
                
                with col_download1:
                    st.download_button(
                        label="Download Annotated Image",
                        data=io_buf,
                        file_name=f"tree_analysis_{int(time.time())}.jpg",
                        mime="image/jpeg",
                    )
                
                # Clean up
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.exception(e)
                if os.path.exists(temp_file):
                    os.remove(temp_file)
    
    # Information section
    st.header("About ArborVision")
    st.markdown("""
    ArborVision is an application that helps estimate tree heights and diameters using computer vision.
    
    **How to use:**
    1. Upload an image containing a tree or take a photo with your camera
    2. The app will automatically detect the tree and calculate its dimensions
    3. You can adjust settings in the sidebar for more precise measurements
    
    **Segmentation methods:**
    - **Auto**: Uses the best available method
    - **Color-based**: Uses color filtering to identify vegetation
    - **Mask R-CNN**: Uses deep learning for object detection (if available)
    - **Shape-based**: Falls back to basic shape detection
    """)

if __name__ == "__main__":
    main()