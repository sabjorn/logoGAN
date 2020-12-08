import os
import shutil

from Gan import Gan
import numpy as np
import sys
import PIL.Image as Image
sys.path.append("../ThirdParty/super-resolution")
from model.srgan import generator
from model import resolve_single

from CVAE import CVAE

from tensorflow import keras

import argparse


def createVideo(contrast, model, latent_dim, output_file="../generated_video/video.mp4", framerate=30, num_cycles=30, n_steps=100, sr_model=None, cleanup_after=False):
    video_dir = os.path.split(output_file)[0]
    frames_dir = os.path.join(video_dir, "frames")
    
    if not os.path.isdir(frames_dir):
        os.mkdir(frames_dir)

    frame_count = 0
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
            image_path = os.path.join(video_dir, "frames", f"test_{str(frame_count).zfill(9)}.png")
            frame_count += 1
            print(f"saving image for cycle {cycle} and vector {i} in {image_path}")
            Image.fromarray(image).save(image_path)

        old_seed = new_seed

    os.system("ffmpeg -y -framerate {0} -i {1}/test_%09d.png -vcodec libx264 {2}".format(framerate, frames_dir, output_file))
    
    # if(cleanup_after):
    #     shutil.rmtree(frames_dir)


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
    parser.add_argument(
        "-k",
        "--keepframes",
        help="flag to keep the generated video frames after creating video",
        action="store_true")
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
    
    createVideo(args.depth, container, latent_dim, num_cycles=args.cycles, n_steps=args.steps, framerate=args.framerate, output_file=args.output, cleanup_after=(not args.keepframes))
