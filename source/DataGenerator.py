import numpy as np
import os
from PIL import Image
from tensorflow import float32, convert_to_tensor
import matplotlib.pyplot as plt


class DataGenerator:
    def __init__(self, imgDims, datasetPath, filetypes=[".jpg", ".jpeg", ".png"]):
        self.selected_data = DataGenerator.create_filelist(datasetPath, filetypes)
        self.imgDims = imgDims
        self.datasetPath = datasetPath
        self.memmapPath = os.path.join(self.datasetPath, 'train.dat')
        self.numFiles = len(os.listdir(datasetPath))
        if not os.path.isfile(self.memmapPath):
            self.createMemmap()

    def createMemmap(self):
        memmap = np.memmap(self.memmapPath, dtype='float32', mode='w+', shape=(*self.imgDims, self.numFiles))
        for n, imgFile in enumerate(self.selected_data):
            print(f'Writing {n}/{self.numFiles} ({imgFile})')
            imgFilePath = os.path.join(self.datasetPath, imgFile)
            memmap[:, :, :, n] = DataGenerator.getProcessedImage(imgFilePath, self.img_dims)
        del memmap
        print('done.')

    def getBatch(self, batchSize):
        memmap = np.memmap(self.memmapPath, dtype='float32', mode='r', shape=(*self.imgDims, self.numFiles))
        indices = np.random.randint(0, self.numFiles-1, size=batchSize)
        imgArrays = []
        for i in indices:
            img = memmap[:, :, :, i]
            imgArrays.append(img)
        batch = np.stack(imgArrays, axis=0)
        batch = convert_to_tensor(batch, dtype=float32)
        del memmap

        return batch

    @staticmethod
    def getProcessedImage(img_file_path, img_dims, convert=None):
        img = Image.open(img_file_path)
        
        if convert:
            img = img.convert(convert)
        
        img = DataGenerator.expand2square(img, 0)
        img = img.resize(img_dims[:2], Image.LANCZOS)

        img_data = np.asarray(img)
        # "L" converted Image is shape == (width, height)
        if len(img_data.shape) is 2:
            img_data = np.expand_dims(img_data, 2)

        # scale [-1, 1]
        img_data = np.divide(img_data, (255 * 0.5))
        img_data = np.subtract(img_data, 1)

        return img_data

    @staticmethod
    def create_filelist(path, filetypes = [".jpg", ".jpeg"]):
        all_files = os.listdir(path)
        filtered_files = set()
        for filetype in filetypes:
            filtered = filter(lambda x: x.endswith(filetype), all_files)
            filtered_files.update(filtered)
        return filtered_files

    @staticmethod
    def expand2square(pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        
        if width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
    
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result 
