import os
import logging

import numpy as np
from PIL import Image
from tensorflow import float32, convert_to_tensor


class DataGenerator:
    def __init__(self, img_dims, datasetPath, filetypes=[".jpg", ".jpeg", ".png"], convert_bw=True):
        self.logger = logging.getLogger(__name__)

        self.convert = None
        if convert_bw:
            self.convert = "L"
        
        self.datasetPath = datasetPath
        
        image_list = DataGenerator.create_filelist(datasetPath, filetypes)
        self.selected_data = DataGenerator.filter_broken_images(image_list)     
        self.numFiles = len(self.selected_data)        
        self.img_dims = img_dims

        self.memmapPath = os.path.join(self.datasetPath, 'train.dat')
        if not os.path.isfile(self.memmapPath):
            self.generate_mmap()

        self.logger.info("done")

    def generate_mmap(self):
        self.logger.info("generating mmap")
        memmap = np.memmap(self.memmapPath, dtype='float32', mode='w+', shape=(*self.img_dims, self.numFiles))
        for n, img_file in enumerate(self.selected_data):
            try:
                img = Image.open(img_file)
                self.logger.info(f'Writing {n}/{self.numFiles} ({img_file})')
            except Exception as e:
                self.logger.error("{0} -- error opening image, skipping {1}".format(e, img_file))
                continue
            memmap[:, :, :, n] = DataGenerator.prepare_image(img, self.img_dims, self.convert)
        del memmap

    def getBatch(self, batchSize):
        memmap = np.memmap(self.memmapPath, dtype='float32', mode='r', shape=(*self.img_dims, self.numFiles))
        indices = np.random.randint(0, self.numFiles-1, size=batchSize)
        imgArrays = []
        for i in indices:
            img = memmap[:, :, :, i]
            imgArrays.append(img)
        batch = np.stack(imgArrays, axis=0)
        batch = convert_to_tensor(batch, dtype=float32)
        del memmap

        return batch

    def __len__(self):
        return self.numFiles

    @staticmethod
    def create_filelist(path, filetypes = [".jpg", ".jpeg"]):
        all_files = os.listdir(path)
        all_files = [os.path.join(path, file) for file in all_files]
        filtered_files = set()
        for filetype in filetypes:
            filtered = filter(lambda x: x.endswith(filetype), all_files)
            filtered_files.update(filtered)
        return filtered_files

    @staticmethod
    def filter_broken_images(image_filenames):
        working_paths = []
        for path in image_filenames:
            try:
                img = Image.open(path)
                working_paths.append(path)
            except Exception as e:
                print("{0} -- error opening image, skipping {1}".format(e, path))
        return working_paths

    @staticmethod
    def add_to_mmap(index_path, memmap, img_dims):
        n, img_file_path = index_path
        try:
            img = Image.open(img_file_path)
            print(f'Writing {n}/{memmap.shape[3]} ({imgFile})')
        except Exception as e:
            print("{0} -- error opening image, skipping {1}".format(e, img_file_path))
            return
        memmap[:, :, :, n] = DataGenerator.prepare_image(img, img_dims)      


    @staticmethod
    def prepare_image(img, img_dims, convert=None):       
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
