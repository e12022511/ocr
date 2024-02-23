import os

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def build_deskew_model(height, width, channels):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, channels)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear'),
    ])
    return model


def organize_data(path):
    images = []
    labels = []
    subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
    for subfolder in subfolders:
        print(f"Current Subfolder: {subfolder}")
        for filename in os.listdir(subfolder):
            image_name = os.path.join(subfolder, filename)
            img = cv2.imread(image_name)
            img = cv2.resize(img, (200, 200))
            img = img / 255.0
            images.append(img)
            labels.append(float(os.path.basename(subfolder)))
    return np.array(images), np.array(labels)


def plot_result(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    images, labels = organize_data(
        os.path.abspath(os.path.join(os.path.dirname(__file__), r'..\resources\sorted_data')))

    split_ratio = 0.8
    split_index = int(len(images) * split_ratio)
    train_images, val_images = images[:split_index], images[split_index:]
    train_labels, val_labels = labels[:split_index], labels[split_index:]

    model = build_deskew_model(200, 200, 3)
    model.compile(optimizer='adam',
                  loss="mse")
    history = model.fit(train_images, train_labels, epochs=20, validation_data=(val_images, val_labels))
    plot_result(history)
    model.save('test5')
