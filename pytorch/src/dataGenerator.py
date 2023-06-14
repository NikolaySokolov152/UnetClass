from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import cv2
import skimage.io as io
import numpy as np
import os
import random
import albumentations as albu
import sys

import time

from torch.utils.data import DataLoader


def to_0_1_format_img(in_img):
    max_val = in_img[:, :].max()
    if max_val <= 1:
        return in_img
    else:
        out_img = in_img / 255
        return out_img

class InfoDirData:
    def __init__(self, dir_img_name="data/train/origin",
                 dir_mask_name="data/train/",
                 add_mask_prefix='mask_'):
        self.dir_img_name = dir_img_name
        self.dir_mask_name = dir_mask_name
        self.add_mask_prefix = add_mask_prefix

    def __str__(self):
        return str(self.__dict__)

class TransformData:
    def __init__(self, color_mode_img='gray',
                 mode_mask='separated',
                 target_size=(256, 256),
                 batch_size=1):
        self.color_mode_img = color_mode_img
        self.mode_mask = mode_mask
        self.target_size = target_size
        self.batch_size = batch_size

    def __str__(self):
        return str(self.__dict__)

class SaveData:
    def __init__(self, save_to_dir=None,
                 save_prefix_image="image_",
                 save_prefix_mask="mask_"):
        self.save_to_dir = save_to_dir
        self.save_prefix_image = save_prefix_image
        self.save_prefix_mask = save_prefix_mask

    def __str__(self):
        return str(self.__dict__)


'''
class TailingData:
    def __init__(self, save_to_dir = None,
                       save_prefix_image="image_",
                       save_prefix_mask="mask_"):
        self.save_to_dir = save_to_dir
        self.save_prefix_image = save_prefix_image
        self.save_prefix_mask = save_prefix_mask
'''


def create_transform(aug_dict, transform_data, augment=True):
    list_compose = []

    if augment:
        if "noise_limit" in aug_dict.keys() and aug_dict["noise_limit"] != 0:
            list_compose.append(albu.GaussNoise(p=0.9, var_limit=aug_dict["noise_limit"] / 256, per_channel=False))

        if "horizontal_flip" in aug_dict.keys() and aug_dict["horizontal_flip"]:
            list_compose.append(albu.HorizontalFlip(p=0.5))
        if "vertical_flip" in aug_dict.keys() and aug_dict["vertical_flip"]:
            list_compose.append(albu.VerticalFlip(p=0.5))

        list_compose.append(
            albu.ShiftScaleRotate(p=0.5, rotate_limit=0, scale_limit=0, shift_limit_x=aug_dict["width_shift_range"], \
                                  shift_limit_y=aug_dict["height_shift_range"], border_mode=aug_dict["fill_mode"]))

        if "brightness_shift_range" in aug_dict.keys() and "contrast_shift_range" in aug_dict.keys():
            list_compose.append(albu.RandomBrightnessContrast(p=0.33,
                                                              brightness_limit=aug_dict["brightness_shift_range"],
                                                              contrast_limit=aug_dict["contrast_shift_range"]))
        if "gamma_limit" in aug_dict.keys():
            list_compose.append(albu.RandomGamma(p=0.33, gamma_limit=aug_dict["gamma_limit"]))

        if "rotation_range" in aug_dict.keys():
            list_compose.append(albu.Rotate(p=0.5, limit=aug_dict["rotation_range"], border_mode=aug_dict["fill_mode"]))

        if "zoom_range" in aug_dict.keys():
            list_compose.append(albu.RandomResizedCrop(p=1, height=transform_data.target_size[1],
                                                       width=transform_data.target_size[0], \
                                                       scale=(1 - aug_dict["zoom_range"], 1 + aug_dict["zoom_range"]),
                                                       ratio=(1 - aug_dict["zoom_range"], 1 + aug_dict["zoom_range"])))

    list_compose.append(
        albu.Resize(height=transform_data.target_size[1], width=transform_data.target_size[0]))

    # add more https://albumentations.ai/docs/api_reference/full_reference/

    return albu.Compose(list_compose)

class Generator(Dataset):
    def __init__(self, dir_data, list_class_name, num_classes, transform_data, aug_dict, mode, save_inform, tailing,
                 augment):
        self.dir_data = dir_data
        self.list_class_name = list_class_name
        self.num_classes = num_classes
        self.transform_data = transform_data
        self.mode = mode
        self.save_inform = save_inform[0]
        self.inform_generator = save_inform[1]
        self.tailing = tailing
        self.augment = augment
        self.batch_size = self.transform_data.batch_size

        self.small_list_img_name = None

        if augment:
            if self.inform_generator == "validation generator":
                print("\nINFO: I'm a validation generator turn off augmentation for myself !\n")
                self.augment = False

        self.composition = create_transform(aug_dict, self.transform_data, self.augment)

        self.first_loop = True


    def __len__(self):
        'Denotes the number of batches per epoch'

        len_gen = int(np.floor(len(self.small_list_img_name) / self.transform_data.batch_size))
        if len_gen == 0:
            print("very small dataset, please add more img")

        return len_gen

    def __getitem__(self, index):
        # if self.inform_generator == "validation generator":
        #    print(index)
        if index >= self.__len__():
            raise StopIteration

        'Generate one batch of data'
        # read_names = self.small_list_img_name[index * self.transform_data.batch_size:(index+1) * self.transform_data.batch_size]
        #                                    len(self.small_list_img_name)]
        read_names = []
        for i in range(self.transform_data.batch_size):
            name = self.small_list_img_name[(index * self.transform_data.batch_size + i) % \
                                            len(self.small_list_img_name)]
            read_names.append(name)

        X, X_shapes = self.generate_X(read_names)
        if self.mode == 'train':
            y = self.generate_Y(read_names, X_shapes)

            X, y = self.augment_batch(X, y)

            X = torch.from_numpy(np.array(X)).permute(0, 3, 1, 2)
            y = torch.from_numpy(np.array(y)).permute(0, 3, 1, 2)
            return X, y

        elif self.mode == 'predict':
            return torch.from_numpy(np.array(X)).permute(0, 3, 1, 2)

        else:
            raise AttributeError('The mode parameter should be set to "train" or "predict".')

    def generate_X(self, list_name_in_batch):
        'Generates data containing batch_size samples'
        # Initialization
        X = []
        list_shape_X = []

        # Generate data
        for i, name in enumerate(list_name_in_batch):
            img_path = os.path.join(self.dir_data.dir_img_name, name)

            if self.transform_data.color_mode_img == "rgb":
                img = io.imread(img_path)
            elif self.transform_data.color_mode_img == "hsv":
                image = io.imread(img_path)
                img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV_FULL)
            else:
                image = io.imread(img_path, as_gray=True)
                img = np.expand_dims(image, axis=-1)

            # print(img.shape)
            img = img.astype(np.float32)
            img = to_0_1_format_img(img)
            # Store samples
            X.append(img)
            list_shape_X.append(img.shape[:2])

        return X, list_shape_X

    def generate_Y(self, list_name_in_batch, X_shapes):
        y = []

        if self.transform_data.mode_mask == 'separated':
            for j, name in enumerate(list_name_in_batch):

                y_arr = np.empty((*X_shapes[j], self.num_classes), np.float32)

                for i in range(self.num_classes):
                    img_path = os.path.join(self.dir_data.dir_mask_name, self.list_class_name[i],
                                            self.dir_data.add_mask_prefix + name)

                    if os.path.isfile(img_path):
                        image = io.imread(img_path, as_gray=True)
                        masks = image.astype(np.float32)
                        masks = to_0_1_format_img(masks)
                    else:
                        if self.first_loop:
                            print("no open ", img_path)
                        masks = np.zeros(X_shapes[j], np.float32)
                    # print(masks.shape)

                    y_arr[:, :, i] = masks

                y.append(y_arr)

        else:
            raise AttributeError('The "mode_mask" parameter should be set to "separated" otherwise not implemented.')

        return y

    def random_transform_one_frame(self, img, masks):
        composed = self.composition(image=img, mask=masks)
        aug_img = composed['image']
        aug_masks = composed['mask']

        return aug_img, aug_masks

    def augment_batch(self, img_batch, masks_batch):
        for i in range(self.transform_data.batch_size):
            img_batch[i], masks_batch[i] = self.random_transform_one_frame(img_batch[i], masks_batch[i])

        return img_batch, masks_batch

    def on_epoch_end(self):
        self.first_loop = False
    #    print()
    ##    print(self.inform_generator)
    ##    print(self.small_list_img_name)
    #    print()

class DataGenerator():
    'Load and generate data'

    def __init__(self, dir_data=InfoDirData(),
                 list_class_name=["1", "2", "3", "4"], num_classes=2, mode='train', tailing=False,
                 aug_dict=dict(), augment=False, shuffle=True, seed=42, subsampling="crossover",
                 transform_data=TransformData(), save_inform=SaveData(), share_validat=0.2, silence_mode=False):

        self.typeGen = "DataGenerator"

        self.dir_data = dir_data
        self.list_class_name = list_class_name
        self.num_classes = num_classes
        self.transform_data = transform_data
        self.aug_dict = aug_dict
        self.mode = mode
        self.subsampling = subsampling

        self.save_inform = save_inform

        self.share_val = share_validat
        self.augment = augment
        self.shuffle = shuffle
        self.seed = seed

        self.tailing = tailing

        self.silence_mode = silence_mode

        img_type = ('.png', '.jpg', '.jpeg')
        self.list_img_name = [name for name in os.listdir(dir_data.dir_img_name) if name.endswith((img_type))]
        print("In normal dir:", len(self.list_img_name), "images", flush=True)

        random.shuffle(self.list_img_name)

        if self.mode == "train":
            self.gen_train = Generator(dir_data, list_class_name, num_classes, transform_data, aug_dict, mode,
                                       [save_inform, "train generator"], tailing, augment)
            self.gen_valid = Generator(dir_data, list_class_name, num_classes, transform_data, aug_dict, mode,
                                       [save_inform, "validation generator"], tailing, augment)

        else:
            self.gen_test = Generator(dir_data, list_class_name, num_classes, transform_data, aug_dict, mode,
                                      [save_inform, "test generator"], tailing, False)

        # self.indexes = [i for i in range(len(self.list_img_name))]

        np.random.seed(self.seed)

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'

        # print(int(len(self.list_img_name) * self.share_val))
        # print(int(len(self.list_train_img_name) / self.transform_data.batch_size))
        # print(int(len(self.list_validation_img_name) / self.transform_data.batch_size))

        len_gen = int(np.floor(len(self.list_img_name) / self.transform_data.batch_size))
        if len_gen == 0:
            print("very small dataset, please add more img")

        return len_gen

    def load_new_data(self, img_names):
        imgs = []
        masks_glob = []

        for name in img_names:
            img_path = os.path.join(self.dir_data.dir_img_name, name)

            if self.transform_data.color_mode_img == "rgb":
                image = io.imread(img_path)
                img = cv2.resize(image, self.transform_data.target_size)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                image = io.imread(img_path, as_gray=True)
                image = np.expand_dims(image, axis=-1)
                img = cv2.resize(image, self.transform_data.target_size)

            # print(img.shape)

            img = img.astype(np.float32)
            img = to_0_1_format_img(img)

            # Store samples
            imgs.append(img)

        if self.transform_data.mode_mask == 'separated':
            for name in img_names:
                class_mask = []
                for i in range(self.num_classes):
                    img_path = os.path.join(self.dir_data.dir_mask_name, self.list_class_name[i],
                                            self.dir_data.add_mask_prefix + name)

                    if os.path.isfile(img_path):
                        image = io.imread(img_path, as_gray=True)
                        masks = image.astype(np.float32)
                        masks = to_0_1_format_img(masks)
                    else:
                        # print("no open ", img_path)
                        masks = np.zeros(self.transform_data.target_size)

                    class_mask.append(masks)
                masks_glob.append(class_mask)
        else:
            raise AttributeError('The "mode_mask" parameter should be set to "separated" otherwise not implemented.')

        return imgs, masks_glob

    def on_epoch_end(self):
        size_val = int(len(self.list_img_name) * self.share_val)
        'Updates indexes after each epoch'
        # print("change generator")

        if self.mode == 'train':
            self.gen_train.small_list_img_name = self.list_img_name[:len(self.list_img_name) - size_val]
            self.gen_valid.small_list_img_name = self.list_img_name[len(self.list_img_name) - size_val:]

            # print(self.indexes[:len(self.list_img_name) - size_val])
            # print(self.indexes[len(self.list_img_name) - size_val:])


        elif self.mode == 'predict':
            self.gen_test.small_list_img_name = self.list_img_name
        else:
            raise AttributeError('The "mode" parameter should be set to "train" or "predict".')

        if self.shuffle is True:
            print("shuffling in the generator")
            # print()

            if self.subsampling == 'random':
                # list_shuffle_indexes_train = self.indexes[:len(self.list_img_name) - size_val]
                # list_shuffle_indexes_test  = self.indexes[len(self.list_img_name) - size_val:]

                # np.random.shuffle(list_shuffle_indexes_train)
                # np.random.shuffle(list_shuffle_indexes_test)

                # self.indexes = list_shuffle_indexes_test + list_shuffle_indexes_train

                list_shuffle_train = self.list_img_name[:len(self.list_img_name) - size_val]
                list_shuffle_test = self.list_img_name[len(self.list_img_name) - size_val:]

                np.random.shuffle(list_shuffle_train)
                np.random.shuffle(list_shuffle_test)

                self.list_img_name = list_shuffle_test + list_shuffle_train


            elif self.subsampling == 'crossover':
                # self.indexes = self.indexes[len(self.list_img_name) - size_val:] + self.indexes[:len(self.list_img_name) - size_val]
                self.list_img_name = self.list_img_name[len(self.list_img_name) - size_val:] + self.list_img_name[:len(
                    self.list_img_name) - size_val]

            else:
                raise AttributeError('The "subsampling" parameter should be set to "random" or "crossover".')

class AugmentGenerator(Dataset):
    def __init__(self, images, masks, transform_data, aug_dict, mode, save_inform, tailing, augment):
        self.transform_data = transform_data
        self.mode = mode
        self.save_inform = save_inform[0]
        self.inform_generator = save_inform[1]
        self.tailing = tailing
        self.augment = augment
        self.batch_size = self.transform_data.batch_size

        self.images = images
        self.masks = masks
        self.indexes = None

        if augment:
            if self.inform_generator == "validation generator":
                print("\nINFO: I'm a validation generator turn off augmentation for myself !\n", flush=True)
                self.augment = False

        self.composition = create_transform(aug_dict, self.transform_data, self.augment)

    def __len__(self):
        'Denotes the number of batches per epoch'

        len_gen = int(np.floor(len(self.indexes) / self.transform_data.batch_size))
        if len_gen == 0:
            print("very small dataset, please add more img")

        return len_gen

    def __getitem__(self, index):
        # if self.inform_generator == "validation generator":
        #print(index)
        if index >= self.__len__():
            raise StopIteration

        # 'Generate one batch of data'
        work_indexes = []
        for i in range(self.transform_data.batch_size):
            work_index = self.indexes[(index * self.transform_data.batch_size + i) % \
                                      len(self.indexes)]
            work_indexes.append(work_index)

        X = []
        for work_index in work_indexes:
            X.append(self.images[work_index])

        if self.mode == 'train':
            y = []
            for work_index in work_indexes:
                y.append(self.masks[work_index])

            X, y = self.augment_batch(X, y)

            X = torch.from_numpy(np.array(X)).permute(0, 3, 1, 2)
            y = torch.from_numpy(np.array(y)).permute(0, 3, 1, 2)
            return X, y

        elif self.mode == 'predict':
            return torch.from_numpy(np.array(X)).permute(0, 3, 1, 2)

        else:
            raise AttributeError('The mode parameter should be set to "train" or "predict".')

    def random_transform_one_frame(self, img, masks):
        composed = self.composition(image=img, mask=masks)
        aug_img = composed['image']
        aug_masks = composed['mask']

        return aug_img, aug_masks

    def augment_batch(self, img_batch, masks_batch):
        for i in range(self.batch_size):
            img_batch[i], masks_batch[i] = self.random_transform_one_frame(img_batch[i], masks_batch[i])

        return img_batch, masks_batch

        #for i in range(self.transform_data.batch_size):
        #    img_batch[i], masks_batch[i] = self.random_transform_one_frame(img_batch[i], masks_batch[i])

        #return img_batch, masks_batch

        # def on_epoch_end(self):
    #    print()
    ##    print(self.inform_generator)
    ##    print(self.small_list_img_name)
    #    print()

class AugmentImages(Dataset):
    def __init__(self, imgs, masks, indexes, aug_dict, transform_data, augment):
        self.imgs=imgs
        self.masks=masks
        self.indexes=indexes
        self.augment=augment
        self.composition = create_transform(aug_dict, transform_data, self.augment)

    def __len__(self):
        return len(self.indexes)

    def __iter__(self):
        for index in self.indexes:
            x, y = self.random_transform_one_frame(self.imgs[index], self.masks[index])
            x = torch.from_numpy(x).permute(2, 0, 1)
            y = torch.from_numpy(y).permute(2, 0, 1)
            yield x, y

    def __getitem__(self, item):
        index = self.indexes[item]
        x, y = self.random_transform_one_frame(self.imgs[index], self.masks[index])
        x = torch.from_numpy(x).permute(2, 0, 1)
        y = torch.from_numpy(y).permute(2, 0, 1)

        return x, y

    def random_transform_one_frame(self, img, masks):
        composed = self.composition(image=img, mask=masks)
        aug_img = composed['image']
        aug_masks = composed['mask']

        return aug_img, aug_masks

class DataGeneratorReaderAll():
    'Generates data with reading all data in RAM'

    def __init__(self, dir_data=InfoDirData(),
                 list_class_name=["1", "2", "3", "4"], num_classes=2, mode='train', tailing=False,
                 aug_dict=dict(), augment=False, shuffle=True, seed=42, subsampling="crossover",
                 transform_data=TransformData(), save_inform=SaveData(), share_validat=0.2, silence_mode=False):
        self.typeGen = "DataGeneratorReaderAll"

        self.dir_data = dir_data
        self.list_class_name = list_class_name
        self.num_classes = num_classes
        self.transform_data = transform_data
        self.aug_dict = aug_dict
        self.mode = mode
        self.subsampling = subsampling

        self.save_inform = save_inform

        self.share_val = share_validat
        self.augment = augment
        self.shuffle = shuffle
        self.seed = seed

        self.tailing = tailing

        self.silence_mode = silence_mode

        img_type = ('.png', '.jpg', '.jpeg')

        if not os.path.exists(dir_data.dir_img_name):
            print("Image path not found!")
            raise AttributeError(f"Image path '{dir_data.dir_img_name}' not found!")

        self.list_img_name = [name for name in os.listdir(dir_data.dir_img_name) if name.endswith((img_type))]
        print("In normal dir:", len(self.list_img_name), "images", flush=True)

        self.indexes = [i for i in range(len(self.list_img_name))]

        self.all_imges, self.all_masks = self.load_new_data(self.list_img_name)

        np.random.seed(self.seed)

        if self.mode == "train":
            self.gen_train = AugmentGenerator(self.all_imges, self.all_masks, transform_data, aug_dict, mode,
                                              [save_inform, "train generator"], tailing, augment)
            self.gen_valid = AugmentGenerator(self.all_imges, self.all_masks, transform_data, aug_dict, mode,
                                              [save_inform, "validation generator"], tailing, augment)

        else:
            self.gen_test = AugmentGenerator(self.all_imges, self.all_masks, transform_data, aug_dict, mode,
                                             [save_inform, "train generator"], tailing, augment)

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'

        len_gen = int(np.floor(len(self.list_img_name) / self.transform_data.batch_size))
        if len_gen == 0:
            print("very small dataset, please add more img")

        # print(int(len(self.list_img_name) * self.share_val))
        # print(int(len(self.list_train_img_name) / self.transform_data.batch_size))
        # print(int(len(self.list_validation_img_name) / self.transform_data.batch_size))

        return len_gen

    def load_new_data(self, img_names):
        imgs = []
        masks_glob = []
        time.sleep(0.2)  # чтобы tqdm не печатал вперед print
        for name in tqdm(img_names, file=sys.stdout, desc='Load slices', disable=self.silence_mode):
            img_path = os.path.join(self.dir_data.dir_img_name, name)

            if self.transform_data.color_mode_img == "rgb":
                image = io.imread(img_path)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = io.imread(img_path, as_gray=True)
                image = np.expand_dims(img, axis=-1)

            # print(img.shape)

            img = image.astype(np.float32)
            img = to_0_1_format_img(img)

            # Store samples
            imgs.append(img)


        if self.transform_data.mode_mask == 'separated':
            for index_name, name in enumerate(tqdm(img_names, file=sys.stdout, desc='Load masks', disable=self.silence_mode)):

                class_mask = np.zeros((*imgs[index_name].shape[:2], self.num_classes), np.float32)

                for i in range(self.num_classes):
                    img_path = os.path.join(self.dir_data.dir_mask_name, self.list_class_name[i],
                                            self.dir_data.add_mask_prefix + name)

                    if os.path.isfile(img_path):
                        image = io.imread(img_path, as_gray=True)
                        masks = image.astype(np.float32)
                        masks = to_0_1_format_img(masks)
                    else:
                        print("no open ", img_path)
                        masks = np.zeros(imgs[index_name].shape[:2], np.float32)

                    class_mask[:, :, i] = masks
                masks_glob.append(class_mask)
        else:
            raise AttributeError('The "mode_mask" parameter should be set to "separated" otherwise not implemented.')

        return imgs, masks_glob

    def getTrainDataLoaderPytorch(self, num_workers = 0):
        gen = AugmentImages(self.all_imges, self.all_masks, self.gen_train.indexes, self.aug_dict, self.transform_data, self.augment)
        smart_gen = DataLoader(gen,
                               batch_size=self.transform_data.batch_size,
                               num_workers=num_workers
                               )
        return smart_gen

    def on_epoch_end(self):
        size_val = int(round(len(self.list_img_name) * self.share_val))
        'Updates indexes after each epoch'
        # print("change generator")

        if self.mode == 'train':
            self.gen_train.indexes = self.indexes[:len(self.list_img_name) - size_val]
            self.gen_valid.indexes = self.indexes[len(self.list_img_name) - size_val:]

            # print(self.indexes[:len(self.list_img_name) - size_val])
            # print(self.indexes[len(self.list_img_name) - size_val:])


        elif self.mode == 'predict':
            self.gen_test.indexes = self.indexes
        else:
            raise AttributeError('The "mode" parameter should be set to "train" or "predict".')

        if self.shuffle is True:
            print("shuffling in the generator")
            # print()

            if self.subsampling == 'random':
                list_shuffle_indexes_train = self.indexes[:len(self.list_img_name) - size_val]
                list_shuffle_indexes_test = self.indexes[len(self.list_img_name) - size_val:]

                np.random.shuffle(list_shuffle_indexes_train)
                np.random.shuffle(list_shuffle_indexes_test)

                self.indexes = list_shuffle_indexes_test + list_shuffle_indexes_train

            elif self.subsampling == 'crossover':
                self.indexes = self.indexes[len(self.list_img_name) - size_val:] + self.indexes[
                                                                                   :len(self.list_img_name) - size_val]

            else:
                raise AttributeError('The "subsampling" parameter should be set to "random" or "crossover".')
