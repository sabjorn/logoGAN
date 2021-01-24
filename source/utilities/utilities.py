import numpy as np
from tensorflow import keras

import sys
sys.path.append("../")
from CVAE import CVAE, Sampling

def interpolate_points(p1, p2, n_steps=100):
    interpolation_plane = np.zeros([n_steps, p1.shape[0]])
    zipped = np.dstack((p1, p2))[0]
    for i, p in enumerate(zipped):
        interpolation_plane[:, i] = np.linspace(p[0], p[1], num=n_steps)
    return interpolation_plane


def load_model(model_file):
    model = keras.models.load_model(model_file)
    
    input_shape = model.output_shape[1:]
    latent_dim = model.input_shape[1]
    
    if("decoder" in model_file):
        print("using CVAE")
        container = CVAE(
              input_shape=input_shape,
              latent_dim=latent_dim)
        container.decoder = model
    
    if("generator" in model_file):
        print("using GAN")
        container = Gan(data_generator=None, input_shape=input_shape, latent_dim=latent_dim)
        container.generator = model

    return container
