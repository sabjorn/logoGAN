import logging
from DataGenerator import DataGenerator
from Gan import Gan

IMG_DIMS = (1024, 1024, 3)
DATA_PATH = "/data"

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

data_generator = DataGenerator(IMG_DIMS, DATA_PATH, filetypes=['.png'], convert_bw=True)
gan = Gan(data_generator=data_generator,
          imgDims=IMG_DIMS,
          batchSize=64,
          noiseDims=100)

gan.train(epochs=10000,
          checkpointFrequency=1000)

