import os
import time
import sys
import json

import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow import keras, function
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam

from DataGenerator import DataGenerator


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class CVAE:
    def __init__(self, data_generator=None, input_shape=(128, 128, 1), batch_size=16, latent_dim=100, seed=None):
        if data_generator:
            self.data_generator = data_generator
            self.batch_size = batch_size
            self.steps_per_epoch = int(len(self.data_generator) / self.batch_size)

        self.latent_dim = latent_dim
        self.input_shape = input_shape

        self.seed = seed

        self.imageSavePath = '../generated_images'
        if not os.path.isdir(self.imageSavePath):
            os.mkdir(self.imageSavePath)
        self.modelSavePath = '../saved_models'
        if not os.path.isdir(self.modelSavePath):
            os.mkdir(self.modelSavePath)

        self.learning_rate = 1e-4
        self.encoder_optimizer = Adam(self.learning_rate)
        self.decoder_optimizer = Adam(self.learning_rate)
        self.encoder = self.create_encoder()
        self.decoder = self.create_decoder()

    def create_encoder(self):
        encoder_inputs = keras.Input(shape=self.input_shape)

        x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(16, activation="relu")(x)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        encoder.summary()

        return encoder

    def create_decoder(self):
        num_layers = 2
        scale = 2
        N = self.input_shape[0] // scale ** num_layers

        latent_inputs = keras.Input(shape=(self.latent_dim,))

        x = layers.Dense(N * N * 64, activation="relu")(latent_inputs)
        x = layers.Reshape((N, N, 64))(x)
        x = layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=1, padding="same")(x)
        x = layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=1, padding="same")(x)
        decoder_outputs = layers.Conv2DTranspose(self.input_shape[2], 3, activation="tanh", padding="same")(x)
        decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        decoder.summary()

        return decoder

    def saveModel(self, epoch):
        self.encoder.save(os.path.join(self.modelSavePath, f"encoder_at_epoch{epoch}.h5"))
        self.decoder.save(os.path.join(self.modelSavePath, f"decoder_at_epoch{epoch}.h5"))

    def save_random_images(self, epoch, num_samples, imageSavePath):
        if(self.seed):
            np.random.seed(self.seed)
        
        z_sample = np.random.normal(-1, 1, (num_samples, self.latent_dim))
        x_decoded = self.decoder.predict(z_sample)

        for i, digit in enumerate(x_decoded):
            pred = digit.reshape(self.input_shape)
            file_path = os.path.join(imageSavePath, f'image_at_epoch_{epoch}_#{i}.png')
            image = np.asarray(pred * 127.5 + 127.5, dtype='uint8')
            Image.fromarray(image).save(file_path)

    def generate_image(self, z_sample):
        x_decoded = self.decoder.predict(z_sample)
        pred = x_decoded.reshape(self.input_shape)
        return np.asarray(pred * 127.5 + 127.5, dtype='uint8')

    def load_encoder(self, filePath):
        self.encoder.load_weights(filePath)

    @function
    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as encoder_tape, tf.GradientTape() as decoder_tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= self.input_shape[0] * self.input_shape[1]
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        
        grads = encoder_tape.gradient(total_loss, self.encoder.trainable_weights)
        self.encoder_optimizer.apply_gradients(zip(grads, self.encoder.trainable_weights))

        grads = decoder_tape.gradient(total_loss, self.decoder.trainable_weights)
        self.decoder_optimizer.apply_gradients(zip(grads, self.decoder.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }


    def train(self, epochs, checkpoint_frequency, num_checkpoint_image=1):
        loss_record = []
        start_time = time.time()

        for epoch in range(epochs):
            start = time.time()

            for batchNum in range(self.steps_per_epoch):
                print(f"EPOCH = {epoch}; BATCH = {batchNum}/{self.steps_per_epoch}")
                image_batch = self.data_generator.getBatch(self.batch_size)
                loss_data = self.train_step(image_batch)
                
                loss_data['loss'] = loss_data['loss'].numpy()
                loss_data['reconstruction_loss'] = loss_data['reconstruction_loss'].numpy()
                loss_data['kl_loss'] = loss_data['kl_loss'].numpy() 
                loss_record.append(loss_record)


            if (epoch + 1) % checkpoint_frequency == 0:
                self.save_random_images(epoch + 1, num_checkpoint_image, self.imageSavePath)
                self.saveModel(epoch + 1)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        self.save_random_images(epochs, num_samples=5, imageSavePath=self.imageSavePath)

        data_output = {
            "start_time": start_time,
            "end_time": time.time(),
            "learning_rate": self.learning_rate,
            "input_shape": self.input_shape,
            "latent_dim": self.latent_dim,
            "loss_record": loss_record
        }
        with open('../experiment.json', 'w') as f:
            json.dump(data_output, f)

