import warnings
warnings.filterwarnings("ignore")
import os
import cv2
import numpy as np
from keras.utils import np_utils
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, Activation

# Set paths and parameters
data_path = "C:/Users/Bavatarinee TM/Downloads/archive (2)/full_df.csv"  # Adjust path as needed
img_size = 256
data = []
target = []

# Load and preprocess data
categories = os.listdir(data_path)
labels = [i for i in range(len(categories))]
label_dict = dict(zip(categories, labels))

for category in categories:
    folder_path = os.path.join(data_path, category)
    img_names = os.listdir(folder_path)

    for img_name in img_names:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        
        try:  
            resized = cv2.resize(img, (img_size, img_size))
            data.append(resized)
            target.append(label_dict[category])
        except Exception as e:
            print('Exception:', e)

# Convert data and target to numpy arrays
data = np.array(data) / 255.0
data = np.reshape(data, (data.shape[0], img_size, img_size, 3))
target = np.array(target)
new_target = np_utils.to_categorical(target)

# Split data into training and test sets
train_data, test_data, train_target, test_target = train_test_split(data, new_target, test_size=0.1)

# Define the model
model = Sequential([
    Conv2D(200, (3, 3), input_shape=data.shape[1:]),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(100, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(128, activation='relu'),
    Dense(256, activation='relu'),
    Dense(len(categories), activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Set up callbacks
checkpoint = ModelCheckpoint("best_model.h5", monitor="accuracy", verbose=1, save_best_only=True)
early_stop = EarlyStopping(monitor="accuracy", min_delta=0.01, patience=5, verbose=1)
callbacks = [checkpoint, early_stop]

# Train the model
hist = model.fit(train_data, train_target, epochs=200, callbacks=callbacks)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_data, test_target, verbose=1)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
print(f'Test Loss: {test_loss:.4f}')

# Plot training accuracy and loss
plt.plot(hist.history['accuracy'])
plt.title("Model Training Accuracy")
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

plt.plot(hist.history['loss'])
plt.title("Model Training Loss")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# Save model architecture
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
