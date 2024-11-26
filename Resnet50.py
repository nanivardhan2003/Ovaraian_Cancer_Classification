#RESNET50
import os
import numpy as np
from zipfile import ZipFile
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.applications import ResNet50
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
    label = 1 # Assign label 1 for cancerous images
    img = Image.open(os.path.join(cancer_dir, filename)).resize((image_width, image_height))
    img_array = img_to_array(img)
    cancer_images.append(img_array)
    cancer_labels.append(label)
# Load non-cancerous tissue images
non_cancer_images = []
non_cancer_labels = []
for filename in os.listdir(non_cancer_dir):
  if filename.endswith(".jpg"):
    label = 0 # Assign label 0 for non-cancerous images
    img = Image.open(os.path.join(non_cancer_dir, filename)).resize((image_width,
    image_height))
    img_array = img_to_array(img)
    non_cancer_images.append(img_array)
    non_cancer_labels.append(label)
# Convert the lists to numpy arrays
cancer_images = np.array(cancer_images)
cancer_labels = np.array(cancer_labels)
non_cancer_images = np.array(non_cancer_images)
non_cancer_labels = np.array(non_cancer_labels)
# Count the number of images in each class
num_cancerous_images = np.sum(cancer_labels == 1)
num_non_cancerous_images = np.sum(non_cancer_labels == 0)
print("Number of cancerous images:", num_cancerous_images)
print("Number of non-cancerous images:", num_non_cancerous_images)
# Display a fixed number of sample images
num_samples = 6 # Set the number of sample images to display
cancerous_indices = np.where(cancer_labels == 1)[0]
non_cancerous_indices = np.where(non_cancer_labels == 0)[0]
if len(cancerous_indices) < num_samples // 2:
  num_samples_cancerous = len(cancerous_indices)
  num_samples_non_cancerous = num_samples - num_samples_cancerous
else:
  num_samples_cancerous = num_samples // 2
  num_samples_non_cancerous = num_samples // 2
  # Randomly select samples from both cancerous and non-cancerous images
cancerous_samples = np.random.choice(cancerous_indices, size=num_samples_cancerous,replace=False)
non_cancerous_samples = np.random.choice(non_cancerous_indices,size=num_samples_non_cancerous, replace=False)
# Display the sample images
for i in range(num_samples_cancerous):
  img = cancer_images[cancerous_samples[i]]
  label = cancer_labels[cancerous_samples[i]]
  label_str = "Cancerous" if label == 1 else "Non-Cancerous"
  plt.imshow(img.astype('uint8'))
  plt.title(label_str)
  plt.axis('off')
  plt.show()
for i in range(num_samples_non_cancerous):
  img = non_cancer_images[non_cancerous_samples[i]]
  label = non_cancer_labels[non_cancerous_samples[i]]
  label_str = "Cancerous" if label == 1 else "Non-Cancerous"
  plt.imshow(img.astype('uint8'))
  plt.title(label_str)
  plt.axis('off')
  plt.show()
# Split the dataset into training and testing sets
X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = train_test_split(
cancer_images, cancer_labels, test_size=0.2, random_state=42)
X_non_cancer_train, X_non_cancer_test, y_non_cancer_train, y_non_cancer_test =train_test_split(non_cancer_images, non_cancer_labels, test_size=0.2, random_state=42)
# Concatenate the cancerous and non-cancerous data
X_train = np.concatenate((X_cancer_train, X_non_cancer_train), axis=0)
y_train = np.concatenate((y_cancer_train, y_non_cancer_train), axis=0)
X_test = np.concatenate((X_cancer_test, X_non_cancer_test), axis=0)
y_test = np.concatenate((y_cancer_test, y_non_cancer_test), axis=0)
unique_labels = np.unique(y_train)
print("Unique Labels:", unique_labels)
# Convert the target labels to one-hot encoded format
num_classes = 2
y_train_encoded = to_categorical(y_train, num_classes=num_classes)
y_test_encoded = to_categorical(y_test, num_classes=num_classes)
# Create an image data generator with data augmentation
datagen = ImageDataGenerator(
rotation_range=20, # Randomly rotate the images
width_shift_range=0.1, # Randomly shift the width of the images
height_shift_range=0.1, # Randomly shift the height of the images
shear_range=0.2, # Apply shearing transformations
zoom_range=0.2, # Apply zoom transformations
horizontal_flip=True, # Randomly flip the images horizontally
vertical_flip=True # Randomly flip the images vertically
)
# Generate augmented training data
augmented_train_data = datagen.flow(X_train, y_train_encoded, batch_size=32)
# Load pre-trained ResNet50 model without the top classification layer
resnet_model = ResNet50(include_top=False, input_shape=(image_height, image_width, 3),
weights='imagenet')
# Freeze the layers of the ResNet50 model
for layer in resnet_model.layers:
  layer.trainable = False
# Create a new model and add the ResNet50 model as a layer
model_resnet = Sequential()
model_resnet.add(resnet_model)
model_resnet.add(Flatten()) # Add a flatten layer to convert feature maps to a 1D feature vector
model_resnet.add(Dropout(0.5)) # Add a dropout layer with rate 0.5
model_resnet.add(Dense(num_classes, activation='softmax',
kernel_regularizer=regularizers.l2(0.01))) # Add a fully connected layer with softmax activation
# Compile the model
model_resnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
# Train the model
history = model_resnet.fit(augmented_train_data, epochs=5, validation_data=(X_test,
y_test_encoded), callbacks=[early_stopping])
# Evaluate the model on the test set
loss, accuracy = model_resnet.evaluate(X_test, y_test_encoded)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
