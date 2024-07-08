import numpy as np
from numpy.fft import fft2, fftshift
# import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageEnhance
import os
from keras.preprocessing.image import img_to_array, load_img

def convert_to_ela_image(path, quality = 90):
    filename = path
    resaved_filename = 'resaved.jpg'
    
    im = Image.open(filename).convert('RGB')
    im.save(resaved_filename, 'JPEG', quality=quality)
    resaved_im = Image.open(resaved_filename)
    
    ela_im = ImageChops.difference(im, resaved_im)
    
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
    # gray_ela = image.convert('L')
    return ela_im

def perform_fft(image):
    image = image.convert('L')  # Convert to grayscale
    f = fft2(np.array(image))
    fshift = fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1)  # Adding 1 to avoid log(0)
    magnitude_spectrum = (magnitude_spectrum - np.min(magnitude_spectrum)) / np.ptp(magnitude_spectrum) * 255
    return Image.fromarray(magnitude_spectrum.astype('uint8'))

def load_images_from_directory(directory, target_size=(128, 128)):
    ela_images = []
    fft_images = []
    labels = []
    for label, category in enumerate(['Real2', 'Fake2']):
        folder_path = os.path.join(directory, category)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
                image = load_img(file_path)
                ela_img = convert_to_ela_image(file_path)
                fft_img = perform_fft(image)
                ela_img = ela_img.resize(target_size)
                fft_img = fft_img.resize(target_size)
                ela_images.append(img_to_array(ela_img) / 255.0)
                fft_images.append(img_to_array(fft_img) / 255.0)
                labels.append(label)  # 0 for real, 1 for fake
    return np.array(ela_images), np.array(fft_images), np.array(labels)

# directory = 'archive/CASIA2'
directory = ''
ela_images, fft_images, labels = load_images_from_directory(directory)
np.save("ela2", ela_images)
np.save("fft2", fft_images)
np.save("labels2", labels)

# ela_image = convert_to_ela_image(imagepath, 90)

# # Convert PIL image to NumPy array
# ela_image_np = np.array(ela_image)

# # Display ELA image using OpenCV
# cv2.imshow("ELA Image", ela_image_np)
# cv2.waitKey(0)
# cv2.destroyAllWindows()