import os

import cv2
import numpy as np
import tensorflow as tf

from deskew_traditional.deskew_method import DeskewMethod


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


if __name__ == '__main__':
    angle = apply_model('test5', cv2.imread(os.path.join(os.path.dirname(__file__), r'..\resources\test\0000054665[2.96].png')))
    print(angle)


class MlMethod(DeskewMethod):

    def deskew(self, path):
        total_path = os.path.join(os.path.dirname(__file__),  r'..', path.replace("/", "\\"))
        rotation_angle = apply_model('deskew_ml/test5', cv2.imread(os.path.join(os.path.dirname(__file__), total_path)))
        return rotation_angle[0][0] * -1
