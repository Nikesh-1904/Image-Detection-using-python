from numpy.fft import fft2, fftshift
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import matplotlib.pyplot as plt

def perform_fft(image):
    image = image.convert('L')  # Convert to grayscale
    f = fft2(np.array(image))
    fshift = fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1)  # Adding 1 to avoid log(0)
    magnitude_spectrum = (magnitude_spectrum - np.min(magnitude_spectrum)) / np.ptp(magnitude_spectrum) * 255
    return Image.fromarray(magnitude_spectrum.astype('uint8'))

image = perform_fft(Image.open('sample/sample6.jpg'))

plt.imshow(image)
plt.title('ELA Image')
plt.axis('off')
plt.show()
