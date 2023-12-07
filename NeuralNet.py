from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras import models
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import data as tf_data
from sklearn.model_selection import train_test_split
from astropy.io import fits 
import os
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix



# -----------------
#   Data
# -----------------


#
#   Preprocessing to VGG19 standard
#
def preprocess_patch(patch):
    # Normalize pixel values (assuming the data range is 0-255)
    patch = (patch - patch.min())/(patch.max()-patch.min())
    return patch

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
    with fits.open(new_image_path) as new_hdul:
        new_image_data = new_hdul[1].data

    with fits.open(ref_image_path) as ref_hdul:
        ref_image_data = ref_hdul[1].data

    for i in range(0, new_image_data.shape[0] - 32, 32):
        for j in range(0, new_image_data.shape[1] - 32, 32):
            new_patch = new_image_data[i:i + 32, j:j + 32]
            ref_patch = ref_image_data[i:i + 32, j:j + 32]

            if new_patch.shape == (32, 32) and ref_patch.shape == (32, 32):
                sub_patch = preprocess_patch(new_patch - ref_patch)
                new_patch = preprocess_patch(new_patch)
                ref_patch = preprocess_patch(ref_patch)

                # Stack and then transpose the combined patch
                combined_patch = np.stack([new_patch, ref_patch, sub_patch], axis=0)
                combined_patch = np.transpose(combined_patch, (1, 2, 0))  # Reshape from (3, 32, 32) to (32, 32, 3)

                has_supernova = check_supernova_in_patch(supernova_locations, j, i)

                image_list.append(combined_patch)
                labels.append(1 if has_supernova else 0)



image_list = []
labels = []

# Process image pairs and populate image_list and labels
# Paths for the first set of images and supernova locations
supernova_locations1 = read_supernova_locations('data/visit_0_3_sn.txt')
process_image_pair('data/visit_0_3_sn_drz.fits', 'data/visit_0_3_orig_drz.fits', supernova_locations1)

# Paths for the second set of images and supernova locations
supernova_locations2 = read_supernova_locations('data/visit_1_2_sn.txt')
process_image_pair('data/visit_1_2_sn_drz.fits', 'data/visit_1_2_orig_drz.fits', supernova_locations2)

# Convert lists to numpy arrays
image_data = np.array(image_list)
label_data = np.array(labels)

# Shuffle and split the data
indices = np.arange(image_data.shape[0])
np.random.shuffle(indices)
image_data = image_data[indices]
label_data = label_data[indices]

# Split the data into training, validation, and test sets
# 70% training, 15% validation, 15% test
train_images, test_images, train_labels, test_labels = train_test_split(
    image_data, label_data, test_size=0.30, random_state=42)
val_images, test_images, val_labels, test_labels = train_test_split(
    test_images, test_labels, test_size=0.50, random_state=42)

# Converting to TensorFlow datasets
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

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
batch_size = 64

train_ds = train_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()
val_ds = val_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()  # Changed from validation_ds to val_ds
test_ds = test_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()



# -----------------------
#   Training Model
# -----------------------

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
model.add(layers.Dropout(0.5))  # Dropout layer to reduce overfitting                                   
model.add(layers.Dense(1, activation='sigmoid'))  # Output layer for binary classification

# Train the model on new data
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
epochs = 100
history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)


# Fine Tuning
# Adjust layers as required
for layer in base_model.layers[-4:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # Low learning rate
    loss='binary_crossentropy',  # Assuming a binary classification (supernova vs no supernova)
    metrics=[tf.keras.metrics.BinaryAccuracy()]
)

epochs_fine_tuning = 100  # Number of epochs for fine-tuning, adjust as needed

history_fine = model.fit(
    train_ds, 
    epochs=epochs_fine_tuning, 
    validation_data=val_ds 
)

# Save model for future use
model.save('supernova_identifier.h5')


# Plot training history
def plot_training_history(history, history_fine):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])  # Initial training
    plt.plot(history.history['val_accuracy'])  # Initial training
    plt.plot(history_fine.history['binary_accuracy'], linestyle='--')  # Fine-tuning
    plt.plot(history_fine.history['val_binary_accuracy'], linestyle='--')  # Fine-tuning
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train Acc', 'Val Acc', 'Train Fine Acc', 'Val Fine Acc'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])  # Initial training
    plt.plot(history.history['val_loss'])  # Initial training
    plt.plot(history_fine.history['loss'], linestyle='--')  # Fine-tuning
    plt.plot(history_fine.history['val_loss'], linestyle='--')  # Fine-tuning
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train Loss', 'Val Loss', 'Train Fine Loss', 'Val Fine Loss'], loc='upper left')
    plt.show()


plot_training_history(history, history_fine)

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(test_ds)
print("Test accuracy:", test_accuracy)
print("Test loss:", test_loss)

# Predict on test data
y_pred = model.predict(test_ds)

print(y_pred)

y_pred = np.round(y_pred)
y_true = np.concatenate([y for x, y in test_ds], axis=0)

print(y_pred)
print(y_true)

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Classification Report
print("Classification Report:\n", classification_report(y_true, y_pred))

