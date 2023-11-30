from keras.applications.vgg19 import VGG19
from keras import models
from keras import layers

from PIL import Image
import numpy as np

#
#   Data
#


#

def read_supernova_locations(filename):
    supernova_locations = []
    with open(filename, 'r') as file:
        for line in file:
            x_center, y_center, flux = line.split()
            supernova_locations.append((float(x_center), float(y_center), float(flux)))
    return supernova_locations

def check_supernova_in_patch(supernova_locations, x_patch, y_patch, patch_size=32):
    for x_center, y_center, _ in supernova_locations:
        if x_patch <= x_center < x_patch + patch_size and y_patch <= y_center < y_patch + patch_size:
            return True
    return False

def process_image_pair(new_image_path, ref_image_path, supernova_locations):
    new_image = Image.open(new_image_path)
    ref_image = Image.open(ref_image_path)

    for i in range(0, new_image.height, 32):
        for j in range(0, new_image.width, 32):
            new_patch = np.array(new_image.crop((j, i, j + 32, i + 32)))
            ref_patch = np.array(ref_image.crop((j, i, j + 32, i + 32)))

            combined_patch = np.stack([new_patch, ref_patch, new_patch - ref_patch], axis=0)

            has_supernova = check_supernova_in_patch(supernova_locations, j, i)

            image_list.append(combined_patch)
            labels.append(1 if has_supernova else 0)

# Example usage
supernova_locations = read_supernova_locations('path_to_your_text_file.txt')
image_list = []
labels = []

process_image_pair('path_to_new_image.jpg', 'path_to_ref_image.jpg', supernova_locations)

# Convert lists to numpy arrays
image_data = np.array(image_list)
label_data = np.array(labels)

import tensorflow as tf

# Assuming image_list and labels are your processed data
dataset = tf.data.Dataset.from_tensor_slices((image_list, labels))

# Shuffling the dataset
dataset = dataset.shuffle(buffer_size=len(image_list))



#

#Cut out 224x224 pixels or 32x32

# Resizing
resize_fn = keras.layers.Resizing(224, 224) #VGG19 requirement 224x224 pixels

train_ds = train_ds.map(lambda x, y: (resize_fn(x), y))
validation_ds = validation_ds.map(lambda x, y: (resize_fn(x), y))
test_ds = test_ds.map(lambda x, y: (resize_fn(x), y))


# Random data augmentation
augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]

def data_augmentation(x):
    for layer in augmentation_layers:
        x = layer(x)
    return x

train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))


# Batch and prefetch
from tensorflow import data as tf_data

batch_size = 64

train_ds = train_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()
validation_ds = validation_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()
test_ds = test_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()


#
#   Training Model
#

# Create base model
base_model = VGG19(weights='imagenet', 
                   input_shape=(32, 32, 3),    #adjust shape
                   include_top=False
) 

# Freeze base model
base_model.trainable = False

# Create a new model
model = models.Sequential()

# Add the VGG19 model
model.add(base_model)

# Add new layers
model.add(layers.Flatten())  # Flatten the output of VGG19
model.add(layers.Dense(256, activation='relu'))  # Add a fully connected layer with 256 neurons
model.add(layers.Dropout(0.5))  # Dropout layer to reduce overfitting                                      #added
model.add(layers.Dense(1, activation='sigmoid'))  # Output layer for binary classification

# Train the model on new data
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=[keras.metrics.BinaryAccuracy()])
epochs = 10
model.fit(train_ds, 
          epochs=epochs, 
          validation_data=validation_ds
)

#
# Fine Tuning
# Adjust layers as required
for layer in base_model.layers[-4:]:
    layer.trainable = True

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
    loss='binary_crossentropy',  # Assuming a binary classification (supernova vs no supernova)
    metrics=[keras.metrics.BinaryAccuracy()]
)

epochs_fine_tuning = 10  # Number of epochs for fine-tuning, adjust as needed

# Continue training with some layers unfrozen
model.fit(
    train_ds, 
    epochs=epochs_fine_tuning, 
    validation_data=validation_ds
)

#
