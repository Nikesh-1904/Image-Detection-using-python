import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

imagepath = 'image2.png'

def detect_edited_areas(image_path, threshold = 45):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute gradients along the x and y axis
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Compute gradient magnitude and normalize
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    grad_magnitude = cv2.normalize(grad_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Threshold to highlight areas with high gradient magnitude
    _, high_grad_regions = cv2.threshold(grad_magnitude, threshold, 255, cv2.THRESH_BINARY)

    # Optional: Dilate the regions to make them more visible
    strec = np.ones((5,5), np.uint8)
    dilated_regions = cv2.dilate(high_grad_regions, strec, iterations=1)
    inverted_regions = cv2.bitwise_not(dilated_regions)
    
    # Find contours in the inverted dilated image
    contours, _ = cv2.findContours(inverted_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour and draw a rectangle around it
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        ss = max(w, h)
        cv2.rectangle(image, (x, y), (x + 2*ss, y + 2*ss), (0, 255, 0), 2)  # Draw rectangle in green
    
    # # Display results
    cv2.imshow('Original Image', image)
    cv2.imshow('Gradient Magnitude', grad_magnitude)
    cv2.imshow('High Gradient Regions', dilated_regions)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
detect_edited_areas(imagepath)



    # # Convert to grayscale to focus on luminance
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Optionally, enhance contrasts or apply further processing here to emphasize light gradients

    # Smooth the image to reduce the impact of minor color variations and noise
    # smoothed = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # # Compute gradients along the x and y axis on the smoothed image
    # grad_x = cv2.Sobel(smoothed, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    # grad_y = cv2.Sobel(smoothed, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    
    # # Compute gradient magnitude-
    # magnitude = cv2.magnitude(grad_x, grad_y)
    
    # # Normalize magnitude for visualization
    # magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # # Threshold to identify significant gradients
    # _, significant_gradients = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
    
    # # Detect areas with significant changes in gradient magnitude
    # # This step could involve morphological operations or additional filtering
    # For simplicity, this example directly uses significant_gradients for visualization

    # Display results
    # cv2.imshow('Original Image', image)
    # cv2.imshow('Gradient Magnitude', magnitude)
    # cv2.imshow('Significant Gradients Detected', significant_gradients)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


