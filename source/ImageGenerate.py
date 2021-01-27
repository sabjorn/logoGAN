""" Generate Random values and use CVAE model and output mapping """
import argparse
import os
import sys

from utilities.utilities import load_model

import numpy as np
import PIL.Image as Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generates a video from a Keras model.")
    parser.add_argument(
        "decoder",
        help="the path to CVAE decoder model",
        type=str)
    parser.add_argument(
        "-o",
        "--output",
        help="output directory",
        type=str)
    parser.add_argument(
        "-s",
        "--seed",
        help="random seed value",
        default=None,
        type=int)
    parser.add_argument(
        "-r",
        "--random",
        help="type of random ('normal' vs 'uniform')",
        default="normal",
        type=str)
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
        "-v",
        "--verbose",
        help="print information about process",
        action="store_true")
    args = parser.parse_args()

    model = load_model(args.decoder)
    latent_dim = model.latent_dim
    input_shape = model.input_shape

    if(args.seed):
        np.random.seed(args.seed)

    if(args.random is "uniform"):
        z_samples = np.random.uniform(-args.scale, args.scale, (args.iter, latent_dim))
    else:
        z_samples = np.random.normal(-1, 1, (args.iter, latent_dim)) * args.scale
    
    try:
        os.mkdir(args.output)
    except FileExistsError:
        pass
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise

    for i, sample in enumerate(z_samples):
        file_path = os.path.join(args.output, f'{args.random}_seed{args.seed}_scale{args.scale}_step{i}.png')

        x_decoded = model.decoder.predict(np.expand_dims(sample, axis=0))
        pred = x_decoded.reshape(input_shape)
        image = np.asarray(pred * 127.5 + 127.5, dtype='uint8')
        Image.fromarray(image).save(file_path)

    if(args.verbose):
        print("lol")