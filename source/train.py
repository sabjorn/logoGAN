import logging
from DataGenerator import DataGenerator
# from Gan import Gan
from CVAE import CVAE

IMG_DIMS = (1024, 1024, 3)
DATA_PATH = "/data"
BACKGROUND_COLOUR = (255, 255, 255)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

data_generator = DataGenerator(IMG_DIMS, DATA_PATH, filetypes=['.png', '.jpg'], use_memmap=False, background_color=BACKGROUND_COLOUR, crop=False, load_from_disk=False)
cvae = CVAE(data_generator=data_generator,
          input_shape=IMG_DIMS,
          batch_size=2,
          latent_dim=16,
          seed=4)

cvae.train(epochs=10000,
          checkpoint_frequency = 1, 
          num_checkpoint_image=5)

