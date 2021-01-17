import logging
from tensorflow import keras
from DataGenerator import DataGenerator
# from Gan import Gan
from CVAE import CVAE, Sampling

IMG_DIMS = (1024, 1024, 3)
DATA_PATH = "/data"
BACKGROUND_COLOUR = (0, 0, 0)
PRETRAINED_ENCODER_PATH = "./8461b9b/saved_models/encoder_at_epoch150.h5"
PRETRAINED_DECODER_PATH = "./8461b9b/saved_models/decoder_at_epoch150.h5"

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

data_generator = DataGenerator(IMG_DIMS, DATA_PATH, filetypes=['.png', '.jpg', '.tif'], use_memmap=False, background_color=BACKGROUND_COLOUR, crop=False, load_from_disk=False)
cvae = CVAE(data_generator=data_generator,
          input_shape=IMG_DIMS,
          batch_size=2,
          latent_dim=16,
          seed=4)

try:
  cvae.encoder = keras.models.load_model(PRETRAINED_ENCODER_PATH, {"Sampling":Sampling})
except NameError as e:
  pass
except:
  print(e)

try:
  cvae.decoder = keras.models.load_model(PRETRAINED_DECODER_PATH)
except NameError as e:
  pass
except:
  print(e)

cvae.train(epochs=150,
          checkpoint_frequency = 10,
          num_checkpoint_image=5)



