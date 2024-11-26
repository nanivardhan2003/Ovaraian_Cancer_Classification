# INCEPTIONV3
import os
import numpy as np
from zipfile import ZipFile
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.applications import InceptionV3
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.callbacks import EarlyStopping
from keras import regularizers
import matplotlib.pyplot as plt

# Set the image dimensions
image_width = 224
image_height = 224

# Load the dataset
dataset_path = 'D:\Capstone\STRAMPN Dataset.zip'
extracted_path = 'D:\Capstone\STRAMPN Dataset'

# Extract the dataset
with ZipFile(dataset_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_path)

# Define the directories for cancerous and non-cancerous images
cancer_dir = os.path.join(extracted_path, 'Dataset', 'Ovarian_Cancer')
non_cancer_dir = os.path.join(extracted_path, 'Dataset', 'Ovarian_Non_Cancer')

# Load cancerous tissue images
cancer_images = []
cancer_labels = []
for filename in os.listdir(cancer_dir):
    if filename.endswith(".jpg"):
        label = 1  # Assign label 1 for cancerous images
        img = Image.open(os.path.join(cancer_dir, filename)).resize((image_width, image_height))
        img_array = img_to_array(img)
        cancer_images.append(img_array)
        cancer_labels.append(label)

# Load non-cancerous tissue images
non_cancer_images = []
non_cancer_labels = []
for filename in os.listdir(non_cancer_dir):
    if filename.endswith(".jpg"):
        label = 0  # Assign label 0 for non-cancerous images
        img = Image.open(os.path.join(non_cancer_dir, filename)).resize((image_width, image_height))
        img_array = img_to_array(img)
        non_cancer_images.append(img_array)
        non_cancer_labels.append(label)

# Convert the lists to numpy arrays
cancer_images = np.array(cancer_images)
cancer_labels = np.array(cancer_labels)
non_cancer_images = np.array(non_cancer_images)
non_cancer_labels = np.array(non_cancer_labels)

# Split the dataset into training and testing sets
X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = train_test_split(
    cancer_images, cancer_labels, test_size=0.2, random_state=42
)
X_non_cancer_train, X_non_cancer_test, y_non_cancer_train, y_non_cancer_test = train_test_split(
    non_cancer_images, non_cancer_labels, test_size=0.2, random_state=42
)

# Concatenate the cancerous and non-cancerous data
X_train = np.concatenate((X_cancer_train, X_non_cancer_train), axis=0)
y_train = np.concatenate((y_cancer_train, y_non_cancer_train), axis=0)
X_test = np.concatenate((X_cancer_test, X_non_cancer_test), axis=0)
y_test = np.concatenate((y_cancer_test, y_non_cancer_test), axis=0)

# Convert the target labels to one-hot encoded format
num_classes = 2
y_train_encoded = to_categorical(y_train, num_classes=num_classes)
y_test_encoded = to_categorical(y_test, num_classes=num_classes)

# Create an image data generator with data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

# Generate augmented training data
augmented_train_data = datagen.flow(X_train, y_train_encoded, batch_size=32)

# Load pre-trained InceptionV3 model without the top classification layer
# Specify weights=None if manually downloading weights
try:
    inception_model = InceptionV3(
        include_top=False,
        input_shape=(image_height, image_width, 3),
        weights='imagenet'  # Use weights=None and specify the manual path if issues persist
    )
except Exception as e:
    print(f"Error loading weights: {e}")
    print("Download the weights manually from the URL and place them in the specified location.")
    # Load manually downloaded weights here:
    weights_path = '/path/to/manual/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
    inception_model = InceptionV3(
        include_top=False,
        input_shape=(image_height, image_width, 3),
        weights=None
    )
    inception_model.load_weights(weights_path)

# Freeze the layers of the InceptionV3 model
for layer in inception_model.layers:
    layer.trainable = False

# Create a new model and add the InceptionV3 model as a layer
model_inception = Sequential()
model_inception.add(inception_model)
model_inception.add(Flatten())
model_inception.add(Dropout(0.5))
model_inception.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.01)))

# Compile the model
model_inception.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history_inception = model_inception.fit(
    augmented_train_data,
    epochs=5,
    validation_data=(X_test, y_test_encoded),
    callbacks=[early_stopping]
)

# Evaluate the model on the test set
loss_inception, accuracy_inception = model_inception.evaluate(X_test, y_test_encoded)
print("Test Loss:", loss_inception)
print("Test Accuracy:", accuracy_inception)
