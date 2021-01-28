""" Generate Random values and use CVAE model and output mapping """
import argparse
import os
import sys

import numpy as np
import PIL.Image as Image

from perlin_numpy import (generate_perlin_noise_2d, generate_perlin_noise_3d)

from tensorflow import keras

from CVAE import Sampling

def perlin_image_input(encoder, res, gain, repeat=(True, True, True)):
    res = res[:3]
    input_shape = encoder.input_shape[1:]
    input_img = generate_perlin_noise_3d(input_shape, res, repeat)
    input_img *= gain
    _, _, z = encoder(np.expand_dims(input_img, 0))
    return z

def perlin_latent_dim(latent_dim, res, gain):
    res = res[0]
    z = generate_perlin_noise_2d((1, latent_dim), (1, res))
    z *= gain
    return z

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generates a video from a Keras model.")
    parser.add_argument(
        "-e",
        "--encoder",
        help="the path to CVAE encoder model",
        type=str)
    parser.add_argument(
        "-d",
        "--decoder",
        help="the path to CVAE decoder model",
        type=str),
    parser.add_argument(
        "-o",
        "--output",
        help="output directory",
        type=str,
        default="./")
    parser.add_argument(
        "-s",
        "--seed",
        help="random seed value",
        default=None,
        type=int)
    parser.add_argument(
        "-a",
        "--scale",
        help="how much to scale the random values by",
        default=1.0,
        type=float),
    parser.add_argument(
        "-i",
        "--iter",
        help="number of images to generate",
        default=1,
        type=int),
    parser.add_argument(
        "-r",
        "--res",
        help="periods of nosie along each axis, note: 'image' take 3 values and 'latent' takes 1",
        nargs="+",
        type=int)
    args = parser.parse_args()

    decoder = keras.models.load_model(args.decoder)
    latent_dim = decoder.input_shape[1]
    input_shape = decoder.output_shape[1:]
    ptype = "latent"

    if args.encoder:
        encoder = keras.models.load_model(args.encoder, {"Sampling":Sampling})
        input_shape = encoder.input_shape[1:]
        ptype = "image"

    outdir = args.output
    try:
        os.makedirs(outdir)
    except:
        pass

    if args.seed: np.random.seed(args.seed)

    gain = args.scale
    num_imgs = args.iter
    res = tuple(args.res)

    for i in range(num_imgs):
        if ptype == "image":
            z = perlin_image_input(encoder, res, gain)
        if ptype == "latent":
            z = perlin_latent_dim(latent_dim, res, gain)

        x_decoded = decoder.predict(z)
        pred = x_decoded.reshape(input_shape)
        image = np.asarray(pred * 127.5 + 127.5, dtype='uint8')

        Image.fromarray(image).save(os.path.join(outdir, f"perlin_type-{ptype}_seed{args.seed}_gain{gain}_res{res}_{i}.png"))
