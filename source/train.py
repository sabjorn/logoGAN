import logging
from DataGenerator import DataGenerator
# from Gan import Gan
from CVAE import CVAE

IMG_DIMS = (128, 128, 3)
DATA_PATH = "/data"

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

data_generator = DataGenerator(IMG_DIMS, DATA_PATH, filetypes=['.png', '.jpg'], use_memmap=False)
cvae = CVAE(data_generator=data_generator,
          input_shape=IMG_DIMS,
          batch_size=128,
          latent_dim=128)

cvae.train(epochs=10000,
          checkpoint_frequency = 100, 
          num_checkpoint_image=1)

