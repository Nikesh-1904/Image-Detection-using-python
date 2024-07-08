import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_image(file_path):
    # Load an image from the file path
    image = Image.open(file_path)
    # Convert the image to grayscale and then to a numpy array
    image = image.convert('L')
    image_array = np.array(image)
    return image_array

def compute_derivatives(image):
    # Calculate the derivative in the y-direction
    dy = np.diff(image, axis=0, prepend=image[:1,:])
    
    # Calculate the derivative in the x-direction
    dx = np.diff(image, axis=1, prepend=image[:,:1])
    
    return dy, dx

def compute_gradient_magnitude(dy, dx):
    # Compute the magnitude of the gradient
    gradient_magnitude = np.sqrt(dy**2 + dx**2)
    return gradient_magnitude

def compute_gradient_direction(dy, dx):
    # Compute the gradient direction using arctan2
    gradient_direction = np.arctan2(dy, dx)
    return gradient_direction

def plot_results(image, dy, dx, gradient_magnitude, gradient_direction):
    # Plotting the original and processed images
    fig, axes = plt.subplots(1, 5, figsize=(24, 6))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(dy, cmap='gray')
    axes[1].set_title('Derivative in Y Direction')
    axes[1].axis('off')

    axes[2].imshow(dx, cmap='gray')
    axes[2].set_title('Derivative in X Direction')
    axes[2].axis('off')

    axes[3].imshow(gradient_magnitude, cmap='gray')
    axes[3].set_title('Gradient Magnitude')
    axes[3].axis('off')

    axes[4].imshow(gradient_direction, cmap='hsv')
    axes[4].set_title('Gradient Direction')
    axes[4].axis('off')

    plt.show()

def plot_histograms(gradient_magnitude, gradient_direction):
    # Convert the direction from radians to degrees for easier interpretation
    gradient_direction_degrees = np.degrees(gradient_direction)

    # Plot histograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Histogram of gradient magnitude
    axes[0].hist(gradient_magnitude.ravel(), bins=50, color='blue', alpha=0.7)
    axes[0].set_title('Histogram of Gradient Magnitude')
    axes[0].set_xlabel('Gradient Magnitude')
    axes[0].set_ylabel('Frequency')

    # Histogram of gradient direction
    axes[1].hist(gradient_direction_degrees.ravel(), bins=50, color='green', alpha=0.7)
    axes[1].set_title('Histogram of Gradient Direction')
    axes[1].set_xlabel('Gradient Direction (degrees)')
    axes[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

file_path = 'sample/sample6.jpg'  # Replace with the path to your image file
image = load_image(file_path)

# Compute derivatives
dy, dx = compute_derivatives(image)

# Compute gradient magnitude
gradient_magnitude = compute_gradient_magnitude(dy, dx)

# Compute gradient direction
gradient_direction = compute_gradient_direction(dy, dx)

# Plot the results
plot_results(image, dy, dx, gradient_magnitude, gradient_direction)

plot_histograms(gradient_magnitude, gradient_direction)
