#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    """Preprocess the image by converting to grayscale, blurring, and thresholding."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Adaptive thresholding to binarize the image
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    return thresh

def segment_fabric(preprocessed_image):
    """Segment the fabric by finding and extracting the largest contour."""
    contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found. Check the image quality.")
    
    # Assume the largest contour corresponds to the fabric
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create a mask for the fabric area
    mask = np.zeros_like(preprocessed_image)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    # Extract the fabric area
    segmented = cv2.bitwise_and(preprocessed_image, preprocessed_image, mask=mask)
    return segmented

def detect_warp_weft_lines_in_cropped(cropped_image):
    """Detect warp and weft lines in the cropped image and color them."""
    # Use Sobel operator to detect edges
    sobel_x = cv2.Sobel(cropped_image, cv2.CV_64F, 1, 0, ksize=5)  # Detect vertical edges (weft)
    sobel_y = cv2.Sobel(cropped_image, cv2.CV_64F, 0, 1, ksize=5)  # Detect horizontal edges (warp)
    
    # Convert the gradients to absolute values
    sobel_x_abs = cv2.convertScaleAbs(sobel_x)
    sobel_y_abs = cv2.convertScaleAbs(sobel_y)
    
    # Create a 3-channel image for colored output
    colored_image = np.zeros((cropped_image.shape[0], cropped_image.shape[1], 3), dtype=np.uint8)

    # Apply mask to color warp lines (horizontal) in red
    colored_image[sobel_y_abs > 50] = [0, 0, 255]  # Red for warp (horizontal)

    # Apply mask to color weft lines (vertical) in blue
    colored_image[sobel_x_abs > 50] = [255, 0, 0]  # Blue for weft (vertical)

    return colored_image

def highlight_individual_warp_weft(cropped_image):
    """
    Detect and highlight individual warp (horizontal) and weft (vertical) threads in fabric.
    - Warp threads (horizontal) → Red
    - Weft threads (vertical) → Blue
    """
    # Apply edge detection using Canny
    edges = cv2.Canny(cropped_image, 50, 150)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=20, maxLineGap=5)

    # Create a blank 3-channel image for visualization
    highlighted_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Determine if the line is horizontal (warp) or vertical (weft)
            if abs(y2 - y1) < abs(x2 - x1):  # Horizontal line (warp)
                color = (0, 0, 255)  # Red
            else:  # Vertical line (weft)
                color = (255, 0, 0)  # Blue

            # Draw the line on the image
            cv2.line(highlighted_image, (x1, y1), (x2, y2), color, thickness=1)

    return highlighted_image

def extract_warp_weft_counts(segmented_image, ppi, original_image):
    """
    Calculate warp and weft counts for 1 square inch and mark the area on the original image.
    """
    # Clean the segmented image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(segmented_image, cv2.MORPH_CLOSE, kernel)
    
    # Calculate size of 1 square inch in pixels
    square_inch_pixels = ppi
    
    # Ensure the image is large enough
    height, width = cleaned.shape
    if height < square_inch_pixels or width < square_inch_pixels:
        raise ValueError("Image does not contain 1 square inch of fabric.")
    
    # Calculate center coordinates
    center_y, center_x = height // 2, width // 2
    half_square = square_inch_pixels // 2

    # Crop 1 square inch from center
    top, bottom = max(0, center_y - half_square), min(height, center_y + half_square)
    left, right = max(0, center_x - half_square), min(width, center_x + half_square)
    cropped = cleaned[top:bottom, left:right]
    
    # Mark the corresponding region on the original image
    marked_image = original_image.copy()
    cv2.rectangle(marked_image, (left, top), (right, bottom), (0, 0, 255), 2)
    
    # Calculate warp and weft counts
    warp_count = np.sum(np.diff(cropped, axis=0) != 0, axis=1).mean()
    weft_count = np.sum(np.diff(cropped, axis=1) != 0, axis=0).mean()
    
    return int(warp_count), int(weft_count), cropped, marked_image

def main(image_path, ppi):
    """Main function to preprocess the image, segment the fabric, and analyze threads."""
    try:
        # Load original image
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Image not found at {image_path}")

        preprocessed = preprocess_image(image_path)
        segmented = segment_fabric(preprocessed)
        warp_count, weft_count, cropped, marked_image = extract_warp_weft_counts(segmented, ppi, original_image)

        # Detect warp and weft lines
        colored_cropped_image = detect_warp_weft_lines_in_cropped(cropped)

        # Highlight individual warp & weft threads
        highlighted_threads = highlight_individual_warp_weft(cropped)

        # Display outputs
        print(f"Warp Count (1 square inch): {warp_count}")
        print(f"Weft Count (1 square inch): {weft_count}")

        plt.figure(figsize=(12, 8))

        plt.subplot(231), plt.imshow(preprocessed, cmap='gray'), plt.title("Preprocessed Image")
        plt.subplot(232), plt.imshow(segmented, cmap='gray'), plt.title("Segmented Image")
        plt.subplot(233), plt.imshow(cropped, cmap='gray'), plt.title("Cropped 1 Square Inch")
        plt.subplot(234), plt.imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB)), plt.title("Marked Original")
        plt.subplot(235), plt.imshow(cv2.cvtColor(colored_cropped_image, cv2.COLOR_BGR2RGB)), plt.title("Warp (Red) & Weft (Blue)")
        plt.subplot(236), plt.imshow(cv2.cvtColor(highlighted_threads, cv2.COLOR_BGR2RGB)), plt.title("Highlighted Threads")

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    image_path = "img5.jpg"
    ppi = 300  # Pixels per inch
    main(image_path, ppi)


# In[ ]:




