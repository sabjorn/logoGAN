import os
import shutil
import sys
import argparse
import subprocess as sp

import numpy as np

sys.path.append("../ThirdParty/super-resolution")
from model.srgan import generator
from model import resolve_single

import PIL.Image as Image
from tensorflow import keras

from CVAE import CVAE
from Gan import Gan

def createVideo(contrast, model, input_shape, latent_dim, output_file="../generated_video/video.mp4", framerate=30, num_cycles=30, n_steps=100, sr_model=None):
    command = [ "ffmpeg",
            '-y', # (optional) overwrite output file if it exists
            '-f', 'rawvideo',
            '-vcodec','rawvideo',
            '-s', f"{input_shape[0]}x{input_shape[1]}", # size of one frame
            '-pix_fmt', 'rgb24',
            '-r', f"{framerate}", # frames per second
            '-i', '-', # The imput comes from a pipe
            '-an', # Tells FFMPEG not to expect any audio
            f"{output_file}" ]
    pipe = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)

    old_seed = np.random.uniform(-contrast, contrast, latent_dim)
    for cycle in range(num_cycles):
        new_seed = np.random.uniform(-contrast, contrast, latent_dim)
        interpolated = interpolate_points(old_seed, new_seed, n_steps=n_steps)

        for i, frame in enumerate(interpolated):
            image = model.generate_image(np.expand_dims(frame, axis=0))
            if(image.ndim > 3):
                image = image[0]
            if(sr_model):
                image = upscale_image(image, sr_model)
            print(f"cycle {cycle}, vector {i}")
            pipe.stdin.write(image.tostring())
        old_seed = new_seed
    
    pipe.terminate()

def interpolate_points(p1, p2, n_steps=100):
    interpolation_plane = np.zeros([n_steps, p1.shape[0]])
    zipped = np.dstack((p1, p2))[0]
    for i, p in enumerate(zipped):
        interpolation_plane[:, i] = np.linspace(p[0], p[1], num=n_steps)
    return interpolation_plane


def upscale_image(image, model, resize=(256, 256)):
    upscaled = resolve_single(model, image)
    upscaled_resized = Image.fromarray(np.asarray(upscaled)).resize(resize)
    upscaled_resized = np.asarray(upscaled_resized)
    upscaled_second = resolve_single(model, upscaled_resized)
    return np.asarray(upscaled_second)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generates a video from a Keras model.")
    parser.add_argument(
        "model",
        help="the path to Keras model",
        type=str)
    parser.add_argument(
        "-d",
        "--depth",
        help="maximum depth of randomness",
        type=float,
        default=1.0)
    parser.add_argument(
        "-c",
        "--cycles",
        help="number of cycles of randomness",
        type=int,
        default=1)
    parser.add_argument(
        "-s",
        "--steps",
        help="number of steps per random cycle",
        type=int,
        default=10)
    parser.add_argument(
        "-f",
        "--framerate",
        help="framerate of output video",
        type=float,
        default=30.)
    parser.add_argument(
        "-o",
        "--output",
        help="output filename of video",
        type=str,
        default="../generated_video/video.mp4")
    args = parser.parse_args()

    model_weights = args.model
    model = keras.models.load_model(model_weights)
    
    input_shape = model.output_shape[1:]
    latent_dim = model.input_shape[1]
    
    if("decoder" in model_weights):
        print("using CVAE")
        container = CVAE(
              input_shape=input_shape,
              latent_dim=latent_dim)
        container.decoder = model
    
    if("generator" in model_weights):
        print("using GAN")
        container = Gan(data_generator=None, imgDims=input_shape, noiseDims=latent_dim)
        container.generator = model    
    
    createVideo(args.depth, container, input_shape, latent_dim, num_cycles=args.cycles, n_steps=args.steps, framerate=args.framerate, output_file=args.output)
