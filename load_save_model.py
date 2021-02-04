# This is a code to download and save vgg16 net
# This file will transfer h5 source into pb for deploying

from keras.applications import VGG16
from keras.layers import Input
import tensorflow as tf


def load_model():
    """ This function is designed to download and save the model as h5 file
    inputs: none
    :return: none
    """
    vgg_model = VGG16(weights='imagenet', input_tensor=Input(shape=(224, 224, 3)))
    vgg_model.save('vgg16.h5')
    print('Done saving VGG')


def save_model():
    """ This function is designed to exporting the model as pd file
        inputs: none
        :return: none
        """
    tf.keras.backend.set_learning_phase(0)
    model = tf.keras.models.load_model('./vgg16.h5')
    export_path = './venv/include/1'
    tf.saved_model.save(model, export_path)

    print('Done exporting')


# run to load and save model.
if __name__ == '__main__':
    # load_model()
    save_model()

