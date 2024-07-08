import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_image(image_path):
    # Load the image file
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    return np.array(img)

def compute_derivatives(image):
    dy = np.diff(image, axis=0, prepend=image[:1,:])
    dx = np.diff(image, axis=1, prepend=image[:,:1])
    return dy, dx

def compute_gradient_magnitude(dy, dx):
    return np.sqrt(dy**2 + dx**2)

def compute_gradient_direction(dy, dx):
    return np.arctan2(dy, dx)

def segment_image(image, c):
    n, m = image.shape
    row_height = n // c
    col_width = m // c
    segments = []
    for i in range(c):
        for j in range(c):
            start_row = i * row_height
            start_col = j * col_width
            # Handle the last segment differently to cover the entire image size
            end_row = start_row + row_height if (i < c-1) else n
            end_col = start_col + col_width if (j < c-1) else m
            segments.append(image[start_row:end_row, start_col:end_col])
    return segments

def plot_histograms(segments, c):
    num_rows = num_cols = c

    # Figure for gradient magnitudes
    fig1, axs1 = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    # Figure for gradient directions
    fig2, axs2 = plt.subplots(num_rows, num_cols, figsize=(15, 15))

    for i in range(c):
        for j in range(c):
            idx = i * c + j
            segment = segments[idx]
            dy, dx = compute_derivatives(segment)
            gradient_magnitude = compute_gradient_magnitude(dy, dx)
            gradient_direction = compute_gradient_direction(dy, dx)
            
            # Plot gradient magnitude
            axs1[i, j].hist(gradient_magnitude.ravel(), bins=30, alpha=0.75)
            axs1[i, j].set_title(f'Segment {idx + 1} Magnitude')
            
            # Plot gradient direction
            axs2[i, j].hist(gradient_direction.ravel(), bins=30, alpha=0.75)
            axs2[i, j].set_title(f'Segment {idx + 1} Direction')
    
    # Adjust layout and display plots
    fig1.tight_layout()
    fig2.tight_layout()
    plt.show()

file_path = 'sample/sample27.jpg'  # Replace with the path to your image file
image = load_image(file_path)
c = 10  # Define the number of segments along one dimension
segments = segment_image(image, c)
plot_histograms(segments, c)