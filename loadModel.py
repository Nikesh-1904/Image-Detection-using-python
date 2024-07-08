import tensorflow as tf
from keras.preprocessing import image
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

#fake photoes
#sample8 gives real
#sample12 gives real

#real photoes
#sample16 is giving fake
#sample21 is giving fake
#sample23 is giving fake

image_path = 'sample/sample3.jpg'

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
    output_path = 'checkImage.jpg'
    ela_im_resized = ela_im.resize((128, 128))
    # Save the modified image to the output directory
    ela_im_resized.save(output_path)

    return output_path

img_path = convert_to_ela_image(image_path)
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Load the model (make sure to replace 'Models/best_model.keras' with the correct path to your model)
model = tf.keras.models.load_model('Models/best_model.keras')

prediction = model.predict(img_array)

# Get the index of the maximum probability
predicted_class_index = np.argmax(prediction, axis=1)[0]

# Dictionary mapping class indices to class labels
class_labels = {0: 'Fake', 1: 'Real'}

# Get the predicted class label
predicted_class_label = class_labels[predicted_class_index]

# Get the probability of the predicted class
probability = prediction[0][predicted_class_index]

# Print the predicted class and its probability
print(f'Predicted class: {predicted_class_label}, Probability: {probability*100:.4f}%')