import os
from Gan import Gan
import numpy as np
import sys
import PIL.Image as Image
sys.path.append("../ThirdParty/super-resolution")
from model.srgan import generator
from model import resolve_single

from CVAE import CVAE

from tensorflow import keras


def createVideo(contrast, model, latent_dim, video_filename="video.mp4", framerate=30, num_cycles=30, n_steps=100, sr_model=None):
    video_dir = "../generated_video"
    frames_dir = os.path.join(video_dir, "frames")
    if not os.path.isdir(video_dir):
        os.mkdir(video_dir)
    if not os.path.isdir(frames_dir):
        os.mkdir(frames_dir)

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
            image_path = os.path.join(video_dir, "frames", f"test_{str((cycle * n_steps) + i).zfill(3)}.png")
            print(f"saving image for cycle {cycle} and vector {i} in {image_path}")
            Image.fromarray(image).save(image_path)

        old_seed = new_seed

    os.system("ffmpeg -y -framerate {0} -i ../generated_video/frames/test_%03d.png -vcodec libx264 ../generated_video/{1}".format(framerate, video_filename))


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
    # model_weights = '/data/decoder_at_epoch3400.h5'
    # model_weights = "/data/c593046/decoder_at_epoch4600.h5"
    model_weights = "/data/46a99c/generator_at_epoch200.h5"
    model = keras.models.load_model(model_weights)
    
    input_shape = model.output_shape[1:]
    latent_dim = model.input_shape[1]
    
    if("decoder" in model_weights):
        print("using CVAE")
        container = CVAE(
              input_shape=input_shape,
              latent_dim=latent_dim)
    
    if("generator" in model_weights):
        print("using GAN")
        container = Gan(data_generator=None, imgDims=input_shape, noiseDims=latent_dim)
    
    container.generator = model
    createVideo(10, container, latent_dim, num_cycles=30, n_steps=10)
