from sklearn.model_selection import train_test_split
# from keras.optimizers.schedules import ExponentialDecay
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import numpy as np
from keras.models import Model
# from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, concatenate, Dense, BatchNormalization
# from keras.regularizers import l2
from keras.optimizers import Adam
from keras.models import Model
import seaborn as sns
from keras.layers import Lambda

# from keras.optimizers import SGD
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten, Conv2D
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf

# import itertools
# from PIL import Image, ImageChops, ImageEnhance

ela_images = np.load("ela.npy")
fft_images = np.load("fft.npy")
labels = np.load("labels.npy")

# Ensure labels are in the correct format
from keras.utils import to_categorical
labels = to_categorical(labels, num_classes=2)

X_ela_train, X_ela_test, X_fft_train, X_fft_test, y_train, y_test = train_test_split(ela_images, fft_images, labels, test_size=0.2, random_state=42)

X_ela_train, X_ela_val, X_fft_train, X_fft_val, y_train, y_val = train_test_split(X_ela_train, X_fft_train, y_train, test_size=0.25, random_state=42)  # This makes validation 20% of the original data


from keras.applications import VGG16
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, concatenate, Dense, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import Model
from keras.layers import Input, Flatten, Lambda, Dense, Dropout, concatenate
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import tensorflow as tf


# from keras.preprocessing.image import ImageDataGenerator
from keras.src.legacy.preprocessing.image import ImageDataGenerator


# def clone_and_rename(model, new_name_suffix):
#     # Clone the model architecture
#     cloned_model = tf.keras.models.clone_model(model)
#     # Rename each layer explicitly
#     for layer in cloned_model.layers:
#         layer._name = f"{layer.name}_{new_name_suffix}"
#     return cloned_model

# original_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Clone and rename models
base_model_ela = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model_fft = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

for i, layer in enumerate(base_model_ela.layers):
    layer._name = 'ela_' + layer.name + f'_{i}'

for i, layer in enumerate(base_model_fft.layers):
    layer._name = 'fft_' + layer.name + f'_{i}'

# Define input layers
input_ela = Input(shape=(128, 128, 3), name='input_ela')
input_fft = Input(shape=(128, 128, 1), name='input_fft')

# Prepare the grayscale image for the color model
x2_prepared = Lambda(lambda x: tf.tile(x, [1, 1, 1, 3]), name='prepare_gray_for_color')(input_fft)

x2_prepared = Lambda(lambda x: tf.tile(x, [1, 1, 1, 3]), name='prepare_gray_for_color')(input_fft)

# Get model outputs
x1 = base_model_ela(input_ela)
x2 = base_model_fft(x2_prepared)

# Flatten the outputs
x1_flattened = Flatten(name='flatten_ela')(x1)
x2_flattened = Flatten(name='flatten_fft')(x2)

# Combine features from both paths
combined = concatenate([x1_flattened, x2_flattened], name='concatenate_features')

# Fully connected layers
z = Dense(256, activation="relu")(combined)
z = Dropout(0.5)(z)
output = Dense(2, activation="softmax")(z)

# with tf.name_scope('dual_input_vgg'):
#     model = Model(inputs=[input_ela, input_fft], outputs=output)

# Define model
model = Model(inputs=[input_ela, input_fft], outputs=output)

# Compile model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint(
    'Models/best_model_dual_input4.keras',  # Path where to save the model
    monitor='val_accuracy',  # Monitor the validation accuracy
    save_best_only=True,    # Only save the model if 'val_accuracy' has improved
    save_weights_only=False,  # Save the entire model, not just the weights
    mode='max',              # Maximize the monitored metric (val_accuracy)
    verbose=1     # The lower the validation loss, the better the model
)

epochs = 15
batch_size = 100

data_gen_args = {
    'rotation_range': 40,
    'width_shift_range': 0.3,
    'height_shift_range': 0.3,
    'shear_range': 0.3,
    'zoom_range': 0.3,
    'horizontal_flip': True,
    'vertical_flip': True,
    'fill_mode': 'nearest'
}

ela_gen = ImageDataGenerator(**data_gen_args)
fft_gen = ImageDataGenerator(**data_gen_args)

# Assuming X_ela_train and X_fft_train are your datasets for ELA and FFT images, respectively
# Ensure y_train is properly one-hot encoded if using categorical_crossentropy

# Define a generator that synchronizes augmentation on both types of images

def dual_input_generator(X1_data, X2_data, y_data, batch_size):
    gen1 = ela_gen.flow(X1_data, y_data, batch_size=batch_size, seed=42)
    gen2 = fft_gen.flow(X2_data, y_data, batch_size=batch_size, seed=42)
    while True:
        X1_batch, y_batch = next(gen1)
        X2_batch, _ = next(gen2)
        yield (X1_batch, X2_batch), y_batch

# Create a tf.data.Dataset object from the generator
output_signature = (
    (tf.TensorSpec(shape=(None, 128, 128, 3), dtype=tf.float32),
     tf.TensorSpec(shape=(None, 128, 128, 3), dtype=tf.float32)),
    tf.TensorSpec(shape=(None, 2), dtype=tf.float32)
)

train_dataset = tf.data.Dataset.from_generator(
    lambda: dual_input_generator(X_ela_train, X_fft_train, y_train, batch_size),
    output_signature=output_signature
)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    lambda: dual_input_generator(X_ela_val, X_fft_val, y_val, batch_size),
    output_signature=output_signature
)
val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

history = model.fit(
    train_dataset,
    steps_per_epoch=len(X_ela_train) // batch_size,
    epochs=50,
    validation_data=val_dataset,
    validation_steps=len(X_ela_val) // batch_size,
    callbacks=[checkpoint]
)

fig, ax = plt.subplots(2, 1, figsize=(10, 10))

# Plot training & validation loss values
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="Validation loss")
ax[0].set_title('Model loss')
ax[0].set_ylabel('Loss')
ax[0].set_xlabel('Epoch')
ax[0].legend(loc='best', shadow=True)

# Plot training & validation accuracy values
ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")  # Corrected key to 'accuracy'
ax[1].plot(history.history['val_accuracy'], color='r', label="Validation accuracy")  # Corrected key to 'val_accuracy'
ax[1].set_title('Model accuracy')
ax[1].set_ylabel('Accuracy')
ax[1].set_xlabel('Epoch')
ax[1].legend(loc='best', shadow=True)

plt.tight_layout()
plt.show()

# Predict the values from the validation dataset
predictions = model.predict({'input_ela': X_ela_test, 'input_fft': X_fft_test})
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

# Confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
#for 256x256
# model = Sequential()

# # Adjusted input_shape for 256x256 images
# model.add(Conv2D(filters=32, kernel_size=(5,5), padding='valid', 
#                  activation='relu', input_shape=(256, 256, 3)))

# model.add(Conv2D(filters=32, kernel_size=(5,5), padding='valid', 
#                  activation='relu'))

# model.add(MaxPool2D(pool_size=(2,2)))

# model.add(Dropout(0.25))

# model.add(Flatten())
# model.add(Dense(256, activation="relu"))
# model.add(Dropout(0.5))
# model.add(Dense(2, activation="softmax"))






###################


# class_labels = {'Fake': 0, 'Real': 1}

# class_0_dir = 'archive/CASIA2/Tp'
# class_1_dir = 'archive/CASIA2/Au'

# X = []  # for storing image data
# Y = []  # for storing label data

# # Generate lists of file paths
# class_0_files = [os.path.join(class_0_dir, file) for file in os.listdir(class_0_dir)]
# class_1_files = [os.path.join(class_1_dir, file) for file in os.listdir(class_1_dir)]

# for file_path in class_0_files:
#     ela_image = Image.open(file_path).resize((128, 128))
#     X.append(np.array(ela_image).flatten() / 255.0)
#     Y.append(class_labels['Fake'])

# for file_path in class_1_files:
#     ela_image = Image.open(file_path).resize((128, 128))
#     X.append(np.array(ela_image).flatten() / 255.0)
#     Y.append(class_labels['Real'])

# X = np.array(X)
# Y = np.array(Y)

# Y = to_categorical(Y, 2)

# X = X.reshape(-1, 128, 128, 3)

# # Split data into training and testing
# X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=5)

# model = Sequential()

# model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'valid', 
#                  activation ='relu', input_shape = (128,128,3)))

# model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'valid', 
#                  activation ='relu'))

# model.add(MaxPool2D(pool_size=(2,2)))

# model.add(Dropout(0.25))

# model.add(Flatten())
# model.add(Dense(256, activation = "relu"))
# model.add(Dropout(0.5))
# model.add(Dense(2, activation = "softmax"))

# model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"]) 
# checkpoint = ModelCheckpoint(
#     'Models/best_model4.keras',  # Path where to save the model
#     save_best_only=True,  # Only save a model if `val_accuracy` has improved
#     monitor='val_accuracy',  # Monitor the validation accuracy
#     mode='max'  # The higher the validation accuracy, the better the model
# ) 



####################
# input_ela = Input(shape=(128, 128, 3), name='input_ela')
# input_fft = Input(shape=(128, 128, 3), name='input_fft')

# input_ela = Input(shape=(128, 128, 3), name='input_ela')
# input_fft = Input(shape=(128, 128, 1), name='input_fft')  # Assuming FFT images are grayscale

# # First path (ELA)
# x1 = Conv2D(32, (5, 5), activation='relu')(input_ela)
# x1 = Conv2D(32, (5, 5), activation='relu')(x1)
# x1 = MaxPooling2D((2, 2))(x1)
# x1 = Dropout(0.25)(x1)
# x1 = Flatten()(x1)

# # Second path (FFT)
# x2 = Conv2D(32, (5, 5), activation='relu')(input_fft)
# x2 = Conv2D(32, (5, 5), activation='relu')(x2)
# x2 = MaxPooling2D((2, 2))(x2)
# x2 = Dropout(0.25)(x2)
# x2 = Flatten()(x2)

# # Combine features from both paths
# combined = concatenate([x1, x2])

# # Fully connected layers
# z = Dense(256, activation="relu")(combined)
# z = Dropout(0.5)(z)
# output = Dense(2, activation="softmax")(z)

# # lr_schedule = ExponentialDecay(
# #     initial_learning_rate=1e-2,
# #     decay_steps=10000,
# #     decay_rate=0.9)
# # optimizer = SGD(learning_rate=lr_schedule)

# # Create the model
# model = Model(inputs=[input_ela, input_fft], outputs=output)
# model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

# Print the model summary to verify its structure
# model.summary()

# Summary of the model
# model.summary()

# input_ela = Input(shape=(128, 128, 3), name='input_ela')
# input_fft = Input(shape=(128, 128, 1), name='input_fft')  # Assuming FFT images are grayscale

# # First path (ELA)
# x1 = Conv2D(32, (5, 5), activation='relu', kernel_regularizer=l2(0.001))(input_ela)
# x1 = BatchNormalization()(x1)
# x1 = Conv2D(32, (5, 5), activation='relu', kernel_regularizer=l2(0.001))(x1)
# x1 = BatchNormalization()(x1)
# x1 = MaxPooling2D((2, 2))(x1)
# x1 = Dropout(0.25)(x1)
# x1 = Flatten()(x1)

# # Second path (FFT)
# x2 = Conv2D(32, (5, 5), activation='relu', kernel_regularizer=l2(0.001))(input_fft)
# x2 = BatchNormalization()(x2)
# x2 = Conv2D(32, (5, 5), activation='relu', kernel_regularizer=l2(0.001))(x2)
# x2 = BatchNormalization()(x2)
# x2 = MaxPooling2D((2, 2))(x2)
# x2 = Dropout(0.25)(x2)
# x2 = Flatten()(x2)

# # Combine features from both paths
# combined = concatenate([x1, x2])

# # Fully connected layers
# z = Dense(256, activation="relu")(combined)
# z = Dropout(0.5)(z)
# output = Dense(2, activation="softmax")(z)

# # Define model
# model = Model(inputs=[input_ela, input_fft], outputs=output)

# # Compile model
# model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])






#########################

# datagen = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# import logging

# def multiple_input_generator(X1, X2, y, batch_size):
#     logging.basicConfig(level=logging.INFO)
#     genX1 = datagen.flow(X1, y, batch_size=batch_size, seed=42)
#     genX2 = datagen.flow(X2, y, batch_size=batch_size, seed=42)
    
#     while True:
#         try:
#             X1i = next(genX1)
#             X2i = next(genX2)
#             yield (tf.convert_to_tensor(X1i[0], dtype=tf.float32), 
#                    tf.convert_to_tensor(X2i[0], dtype=tf.float32)), \
#                   tf.convert_to_tensor(X1i[1], dtype=tf.float32)
#         except Exception as e:
#             logging.error("Error in generator: ", exc_info=True)
#             break


# # Define output_signature matching the new generator output
# output_signature = (
#     (
#         tf.TensorSpec(shape=(None, 128, 128, 3), dtype=tf.float32),  # for ELA images
#         tf.TensorSpec(shape=(None, 128, 128, 1), dtype=tf.float32)   # for FFT images
#     ),
#     tf.TensorSpec(shape=(None, 2), dtype=tf.float32)  # for labels
# )

# # Create the dataset from the generator
# train_dataset = tf.data.Dataset.from_generator(
#     generator=lambda: multiple_input_generator(X_ela_train, X_fft_train, y_train, batch_size),
#     output_signature=output_signature
# )

# val_dataset = tf.data.Dataset.from_generator(
#     generator=lambda: multiple_input_generator(X_ela_val, X_fft_val, y_val, batch_size),
#     output_signature=output_signature
# )

# history = model.fit(
#     train_dataset,
#     steps_per_epoch=len(X_ela_train) // batch_size,
#     epochs=epochs,
#     validation_data=val_dataset,
#     # batch_size = batch_size,
#     callbacks=[checkpoint]
# )