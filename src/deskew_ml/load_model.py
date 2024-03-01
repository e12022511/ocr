import os

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from src.deskew_traditional.deskew_method import DeskewMethod


def load_model(name):
    return tf.keras.models.load_model(name)


def apply_model(model_name, image):
    model = load_model(model_name)
    image = preprocess_image(image)
    return model.predict(image)


def preprocess_image(image):
    new_image = cv2.resize(image, (200, 200))
    new_image = new_image / 255.0  # Normalize pixel values to [0, 1]

    # Reshape the image to match the input shape expected by the model
    return np.reshape(new_image, (1, 200, 200, 3))


def plot_result(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    model = load_model('model')
    history = model.history
    plot_result(history)


class MlMethod(DeskewMethod):

    def deskew(self, path):
        total_path = os.path.join(os.path.dirname(__file__), r'../..', path.replace("/", "\\"))
        rotation_angle = apply_model('src/deskew_ml/model',
                                     cv2.imread(os.path.join(os.path.dirname(__file__), total_path)))
        return rotation_angle[0][0] * -1
