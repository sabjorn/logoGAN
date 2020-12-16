import os
import glob
import logging

import numpy as np
from PIL import Image
from tensorflow import float32, convert_to_tensor


class DataGenerator:
    def __init__(self, img_dims, datasetPath, filetypes=[".jpg", ".jpeg", ".png"], use_memmap=True, background_color=(0, 0, 0), crop=False, load_from_disk=False):
        self.logger = logging.getLogger(__name__)

        self.background_color = background_color
        self.crop = crop

        self.use_memmap = use_memmap
        self.datasetPath = datasetPath
        
        image_list = DataGenerator.create_filelist(datasetPath, filetypes)
        self.selected_data = DataGenerator.filter_broken_images(image_list)
        self.numFiles = len(self.selected_data)
        self.logger.info("total images to process: {0}".format(self.numFiles))
        
        self.img_dims = img_dims
        self.convert = None
        if (self.img_dims[2] == 1):
            self.convert = "L"

        self.load_from_disk = load_from_disk # bypass loading images

        self.memmapPath = os.path.join(self.datasetPath, 'train.dat')
        if self.use_memmap:
            if not os.path.isfile(self.memmapPath):
                self.generate_mmap()
        else:
            self.numpy_img_path = os.path.join(self.datasetPath, "{0}_{1}_{2}".format(*self.img_dims))
            if not os.path.exists(self.numpy_img_path):
                os.makedirs(self.numpy_img_path)
            self.generate_files()

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
            memmap[:, :, :, n] = DataGenerator.prepare_image(img, self.img_dims, self.convert, background_color=self.background_color)
        del memmap

    def generate_files(self):
        self.logger.info("generating images on disk")
        if not self.load_from_disk:
            for n, img_file in enumerate(self.selected_data):
                img_name = os.path.split(img_file)[1]
                
                glob_image_path = os.path.join(self.numpy_img_path, f"{img_name}*.npy")
                images = glob.glob(glob_image_path)
                
                if(len(images)):
                    self.logger.info("file(s) exists, skipping")
                    continue
                
                try:
                    img = Image.open(img_file)
                    self.logger.info(f'Writing {n}/{self.numFiles} ({img_file})')
                except Exception as e:
                    self.logger.error("{0} -- error opening image, skipping {1}".format(e, img_file))
                    continue

                if self.crop:
                    imgs = DataGenerator.crop_images(img, self.img_dims)
                    for i, sub_img in enumerate(imgs):
                        complete_img_path = os.path.join(self.numpy_img_path, img_name + f"_{i}.npy")
                        np.save(complete_img_path, sub_img)
                else:
                    complete_img_path = os.path.join(self.numpy_img_path, img_name + ".npy")
                    np.save(complete_img_path, DataGenerator.prepare_image(img, self.img_dims, self.convert, background_color=self.background_color))

        self.selected_data = DataGenerator.create_filelist(self.numpy_img_path, [".npy"])
        self.numFiles = len(self.selected_data)

    def getBatch(self, batchSize):
        indices = np.random.randint(0, self.numFiles-1, size=batchSize)
        imgArrays = []
        if(self.use_memmap):
            memmap = np.memmap(self.memmapPath, dtype='float32', mode='r', shape=(*self.img_dims, self.numFiles))
            for i in indices:
                img = memmap[:, :, :, i]
                imgArrays.append(img)
            batch = np.stack(imgArrays, axis=0)
            batch = convert_to_tensor(batch, dtype=float32)
            del memmap
            return batch

        for i in indices:
            img = np.load(self.selected_data[i])
            imgArrays.append(img)
        batch = np.stack(imgArrays, axis=0)
        batch = convert_to_tensor(batch, dtype=float32)
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
        return list(filtered_files)

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
    def prepare_image(img, img_dims, convert=None, background_color=0):       
        if convert:
            img = img.convert(convert)
        
        img = DataGenerator.expand2square(img, background_color)
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
    def crop_images(img, crop_size, convert=None):
        if convert:
            img = img.convert(convert)

        img = np.asarray(img)
        x_steps = img.shape[0]//crop_size[0]
        y_steps = img.shape[1]//crop_size[1]

        imgs = []
        for x_step in range(x_steps):
            x_index = crop_size[0] * x_step
            for y_step in range(y_steps):
                y_index = crop_size[1] * y_step
                
                img_data = img[x_index:x_index+crop_size[0], y_index:y_index+crop_size[1], :crop_size[2]]
                
                # scale [-1, 1]
                img_data = np.divide(img_data, (255 * 0.5))
                img_data = np.subtract(img_data, 1)
                imgs.append(img_data)

        return imgs

    @staticmethod
    def expand2square(pil_img, background_color=0):
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
