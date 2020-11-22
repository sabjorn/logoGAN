import logging
from DataGenerator import DataGenerator
from Gan import Gan

IMG_DIMS = (512, 512, 3)
DATA_PATH = "/data"

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

data_generator = DataGenerator(IMG_DIMS, DATA_PATH, filetypes=['.png'], use_memmap=False)
gan = Gan(data_generator=data_generator,
          imgDims=IMG_DIMS,
          batchSize=64,
          noiseDims=100)

gan.train(epochs=10000,
          checkpointFrequency=100)

