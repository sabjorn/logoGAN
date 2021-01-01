""" Load image into CVAE model and output mapping """
import argparse

import numpy as np
import PIL.Image as Image
from tensorflow import keras

from DataGenerator import DataGenerator
from CVAE import CVAE, Sampling

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generates a video from a Keras model.")
    parser.add_argument(
        "encoder",
        help="the path to CVAE encoder model",
        type=str)
    parser.add_argument(
        "decoder",
        help="the path to CVAE decoder model",
        type=str)
    parser.add_argument(
        "-i",
        "--input",
        help="input iamge",
        type=str)
    parser.add_argument(
        "-o",
        "--output",
        help="output filename for image export",
        type=str)
    parser.add_argument(
        "-j",
        "--joined",
        help="output original image next to processed image",
        action="store_true")
    parser.add_argument(
        "-v",
        "--verbose",
        help="print information about process",
        action="store_true")
    args = parser.parse_args()

    encoder = keras.models.load_model(args.encoder, {"Sampling":Sampling})
    decoder = keras.models.load_model(args.decoder)

    input_dims = encoder.input_shape[1:]

    input_img = Image.open(args.input)
    input_img = DataGenerator.prepare_image(input_img, input_dims, convert=None, background_color=(255, 255, 255))

    z_mean, z_log_var, z = encoder(np.expand_dims(input_img, 0))
    x_decoded = decoder.predict(z)

    if(args.verbose):
        print(f"converting input image: {args.input}")
        print(f"generated parameters: {z}")

    pred = x_decoded.reshape(input_dims)
    img = np.asarray(pred * 127.5 + 127.5, dtype='uint8')

    if(args.joined):
        input_img = np.asarray(input_img * 127.5 + 127.5, dtype='uint8')
        img = np.hstack((input_img, img))

    Image.fromarray(img).save(args.output)
