# DENSENET121
# Import necessary libraries
import os
import numpy as np
from zipfile import ZipFile
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.applications import DenseNet121
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
cancer_dir = os.path.join(extracted_path, 'D:\Capstone\STRAMPN Dataset\Dataset\Ovarian_Cancer')
non_cancer_dir = os.path.join(extracted_path, 'D:\Capstone\STRAMPN Dataset\Dataset\Ovarian_Non_Cancer')
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

# Count the number of images in each class
print("Number of cancerous images:", len(cancer_labels))
print("Number of non-cancerous images:", len(non_cancer_labels))

# Display a fixed number of sample images
num_samples = 6
cancerous_indices = np.where(cancer_labels == 1)[0]
non_cancerous_indices = np.where(non_cancer_labels == 0)[0]

num_samples_cancerous = min(len(cancerous_indices), num_samples // 2)
num_samples_non_cancerous = min(len(non_cancerous_indices), num_samples // 2)

cancerous_samples = np.random.choice(cancerous_indices, size=num_samples_cancerous, replace=False)
non_cancerous_samples = np.random.choice(non_cancerous_indices, size=num_samples_non_cancerous, replace=False)

for i in range(num_samples_cancerous):
    img = cancer_images[cancerous_samples[i]]
    plt.imshow(img.astype('uint8'))
    plt.title("Cancerous")
    plt.axis('off')
    plt.show()

for i in range(num_samples_non_cancerous):
    img = non_cancer_images[non_cancerous_samples[i]]
    plt.imshow(img.astype('uint8'))
    plt.title("Non-Cancerous")
    plt.axis('off')
    plt.show()

# Split the dataset into training and testing sets
X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = train_test_split(
    cancer_images, cancer_labels, test_size=0.2, random_state=42)
X_non_cancer_train, X_non_cancer_test, y_non_cancer_train, y_non_cancer_test = train_test_split(
    non_cancer_images, non_cancer_labels, test_size=0.2, random_state=42)

# Concatenate the data
X_train = np.concatenate((X_cancer_train, X_non_cancer_train), axis=0)
y_train = np.concatenate((y_cancer_train, y_non_cancer_train), axis=0)
X_test = np.concatenate((X_cancer_test, X_non_cancer_test), axis=0)
y_test = np.concatenate((y_cancer_test, y_non_cancer_test), axis=0)

# One-hot encode the labels
num_classes = 2
y_train_encoded = to_categorical(y_train, num_classes=num_classes)
y_test_encoded = to_categorical(y_test, num_classes=num_classes)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)
augmented_train_data = datagen.flow(X_train, y_train_encoded, batch_size=32)

# Load DenseNet121
densenet_model = DenseNet121(include_top=False, input_shape=(image_height, image_width, 3), weights='imagenet')

# Freeze layers
for layer in densenet_model.layers:
    layer.trainable = False

# Build the model
model_densenet = Sequential([
    densenet_model,
    Flatten(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.01))
])

# Compile the model
model_densenet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history_densenet = model_densenet.fit(
    augmented_train_data,
    epochs=5,
    validation_data=(X_test, y_test_encoded),
    callbacks=[early_stopping]
)

# Save the model
model_densenet.save('/content/drive/MyDrive/cancer_detection_model.h5')

# Plot training results
def plot_results(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.show()

# Plot results
plot_results(history_densenet)
