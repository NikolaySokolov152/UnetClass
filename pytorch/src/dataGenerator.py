from torch.utils.data import Dataset
import torch
from torchvision import tv_tensors
from tqdm import tqdm
import cv2
import skimage.io as io
import numpy as np
import os
import random
import sys
import json

from augmentation import *

import time

from torch.utils.data import DataLoader

img_type = ('.png', '.jpg', '.jpeg')

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
                 add_mask_prefix='mask_',
                 proportion_of_dataset = 1.0):
        self.dir_img_name = dir_img_name
        self.dir_mask_name = dir_mask_name
        self.add_mask_prefix = add_mask_prefix
        self.proportion_of_dataset = proportion_of_dataset

    def __str__(self):
        return f"\nInfoDirData:\n\t dir_img_name: {self.dir_img_name}" +\
               f"\n\t dir_mask_name: {self.dir_mask_name}" + \
               f"\n\t add_mask_prefix: {self.add_mask_prefix}" + \
               f"\n\t proportion_of_dataset: {self.proportion_of_dataset}"

    def __repr__(self):
        return self.__str__()

class TransformData:
    def __init__(self, color_mode_img='gray',
                 mode_mask='separated',
                 target_size=(256, 256),
                 batch_size=1,
                 binary_mask=True):
        self.color_mode_img = color_mode_img
        self.mode_mask = mode_mask
        self.target_size = target_size
        self.batch_size = batch_size
        self.binary_mask = binary_mask

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

class AugmentGenerator(Dataset):
    def __init__(self, images, masks, transform_data, aug_dict, mode, save_inform, tailing, augment, device):
        self.transform_data = transform_data
        self.mode = mode
        self.save_inform = save_inform[0]
        self.inform_generator = save_inform[1]
        self.tailing = tailing
        self.augment = augment
        self.batch_size = self.transform_data.batch_size
        self.device = device

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
            X = torch.stack(X)
            y = torch.stack(y)
            return X, y

        elif self.mode == 'predict':
            X = torch.stack(X)
            return X

        else:
            raise AttributeError('The mode parameter should be set to "train" or "predict".')

    def random_transform_one_frame(self, img, masks):
        aug_img, aug_masks = self.composition(tv_tensors.Image(img),
                                              tv_tensors.Mask(masks))
        if self.transform_data.binary_mask:
            aug_masks = aug_masks.round()
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
    def __init__(self, imgs, masks, indexes, aug_dict, transform_data, augment, device):
        self.imgs=imgs
        self.masks=masks
        self.indexes=indexes
        self.augment=augment
        self.composition = create_transform(aug_dict, transform_data, self.augment)
        self.device = device

    def __len__(self):
        return len(self.indexes)

    def __iter__(self):
        for index in self.indexes:
            x, y = self.random_transform_one_frame(self.imgs[index], self.masks[index])
            yield x, y

    def __getitem__(self, item):
        index = self.indexes[item]
        x, y = self.random_transform_one_frame(self.imgs[index], self.masks[index])
        return x, y

    def random_transform_one_frame(self, img, masks):
        ret_img, ret_masks = self.composition(img, masks)
        return ret_img, ret_masks

class DataGeneratorReaderAll():
    'Generates data with reading all data in RAM'

    def __init__(self,
                 dir_data=InfoDirData(),
                 list_class_name=None,
                 num_classes=2,
                 mode='train',
                 tailing=False,
                 aug_dict=dict(),
                 augment=False,
                 shuffle=True,
                 type_load_data="img",
                 seed=42,
                 subsampling="crossover",
                 transform_data=TransformData(),
                 save_inform=SaveData(),
                 share_validat=0.2,
                 silence_mode=False,
                 device = "cuda"):
        self.typeGen = "DataGeneratorReaderAll"

        if list_class_name is None:
            self.list_class_name = [f"{i}" for i in range(num_classes)]
        else:
            self.list_class_name = list_class_name

        if device == 'cuda' and not torch.cuda.is_available():
            raise Exception("Augmentator don't use GPU device !")
        self.device = device

        self.dir_data = dir_data
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


        self.list_img_name = []

        if type_load_data == "img":
            self.all_imges, self.all_masks = self.load_new_data()
        elif type_load_data == "npy":
            self.all_imges, self.all_masks = self.loadNpyData()
        else:
            raise Exception("Now you can choose between loading by images or using an .npy file")

        self.all_imges = self.get_device_tensor_from_list_of_numpy(self.all_imges)
        self.all_masks = self.get_device_tensor_from_list_of_numpy(self.all_masks)

        self.indexes = [i for i in range(len(self.list_img_name))]
        print("In normal dir:", len(self.list_img_name), "images", flush=True)

        np.random.seed(self.seed)

        if self.mode == "train":
            self.gen_train = AugmentGenerator(self.all_imges, self.all_masks, transform_data, aug_dict, mode,
                                              [save_inform, "train generator"], tailing, augment, device)
            self.gen_valid = AugmentGenerator(self.all_imges, self.all_masks, transform_data, aug_dict, mode,
                                              [save_inform, "validation generator"], tailing, augment, device)

        else:
            self.gen_test = AugmentGenerator(self.all_imges, self.all_masks, transform_data, aug_dict, mode,
                                             [save_inform, "test generator"], tailing, augment, device)

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

    def load_new_data(self):
        imgs = []
        masks_glob = []

        if type(self.dir_data) is InfoDirData:
            list_InfoDirData = [self.dir_data]
        elif type(self.dir_data) is list or type(self.dir_data) is tuple:
            list_InfoDirData = self.dir_data
        else:
            raise Exception(f"ERROR type dir_data don't know: {self.dir_data}")

        for i, info_dir_data in enumerate(list_InfoDirData):
            dataset_imgs = []
            dataset_masks_glob = []


            if not self.silence_mode and len(list_InfoDirData) > 1:
                print(f"Load {i+1} dataset of {len(list_InfoDirData)}")
            if not os.path.exists(info_dir_data.dir_img_name):
                print("Image path not found!")
                raise AttributeError(f"Image path '{info_dir_data.dir_img_name}' not found!")

            img_names_all = [name for name in os.listdir(info_dir_data.dir_img_name) if name.endswith((img_type))]
            len_names = len(img_names_all)

            if info_dir_data.proportion_of_dataset < 1:
                work_num_img = int(round(len_names * info_dir_data.proportion_of_dataset))
                if not self.silence_mode:
                    print(
                        f"I take {work_num_img} of {len_names} imagess, proportion of dataset: {info_dir_data.proportion_of_dataset}")
                img_names = random.sample(img_names_all, work_num_img)

            else:
                img_names = img_names_all

            self.list_img_name += img_names
            time.sleep(0.2)  # чтобы tqdm не печатал вперед print
            for name in tqdm(img_names, file=sys.stdout, desc='\tLoad slices', disable=self.silence_mode):
                img_path = os.path.join(info_dir_data.dir_img_name, name)

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
                dataset_imgs.append(img)

            if self.transform_data.mode_mask == 'separated':
                for index_name, name in enumerate(tqdm(img_names, file=sys.stdout, desc='\tLoad masks', disable=self.silence_mode)):

                    class_mask = np.zeros((*dataset_imgs[index_name].shape[:2], self.num_classes), np.float32)

                    for i in range(self.num_classes):
                        img_path = os.path.join(info_dir_data.dir_mask_name, self.list_class_name[i],
                                                info_dir_data.add_mask_prefix + name)

                        if os.path.isfile(img_path):
                            mask_read = io.imread(img_path, as_gray=True)
                            masks = mask_read.astype(np.float32)
                            masks = to_0_1_format_img(masks)
                            if self.transform_data.binary_mask:
                                masks[masks<0.5]=0.0
                                masks[masks>0.4]=1.0
                        else:
                            print("no open ", img_path)
                            masks = np.zeros(dataset_imgs[index_name].shape[:2], np.float32)

                        class_mask[:, :, i] = masks
                    dataset_masks_glob.append(class_mask)

            elif self.transform_data.mode_mask == 'image':
                for index_name, name in enumerate(
                        tqdm(img_names, file=sys.stdout, desc='\tLoad res image', disable=self.silence_mode)):
                    img_path = os.path.join(info_dir_data.dir_mask_name, self.list_class_name[0],
                                            info_dir_data.add_mask_prefix + name)
                    if os.path.isfile(img_path):
                        if self.transform_data.color_mode_img == "rgb":
                            img = io.imread(img_path)
                        elif self.transform_data.color_mode_img == "hsv":
                            image = io.imread(img_path)
                            img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV_FULL)
                        else:
                            image = io.imread(img_path, as_gray=True)
                            img = np.expand_dims(image, axis=-1)

                        masks = img.astype(np.float32)
                        masks = to_0_1_format_img(masks)
                    else:
                        print("no open ", img_path)
                        raise Exception(f"ERROR! No open: {img_path}")
                    # print(masks.shape)
                    dataset_masks_glob.append(masks)

            elif self.transform_data.mode_mask == 'no_mask':
                # заглушка
                for index_name, name in enumerate(
                        tqdm(img_names, file=sys.stdout, desc='Create fake masks', disable=self.silence_mode)):
                    dataset_masks_glob.append(np.zeros((*dataset_imgs[index_name].shape[:2],1), np.float32))
            else:
                raise AttributeError('The "mode_mask" parameter should be set to "separated", "image" or "no_mask" otherwise not implemented.')

            imgs += dataset_imgs
            masks_glob += dataset_masks_glob

        return imgs, masks_glob

    def loadNpyData(self):
        imgs = []
        masks_glob = []

        if type(self.dir_data) is InfoDirData:
            list_InfoDirData = [self.dir_data]
        elif type(self.dir_data) is list or type(self.dir_data) is tuple:
            list_InfoDirData = self.dir_data
        else:
            raise Exception(f"ERROR type dir_data don't know: {self.dir_data}")

        for i, info_dir_data in enumerate(list_InfoDirData):
            if not self.silence_mode and len(list_InfoDirData) > 1:
                print(f"Load {i+1} dataset of {len(list_InfoDirData)}")
            if not os.path.exists(info_dir_data.dir_img_name):
                print("Image path not found!")
                raise AttributeError(f"Image path '{info_dir_data.dir_img_name}' not found!")

            open_img_zip = np.load(info_dir_data.dir_img_name, allow_pickle=True)

            img_names_all = open_img_zip[0]
            len_names = len(img_names_all)
            indexes = [i for i in range(len_names)]

            if info_dir_data.proportion_of_dataset < 1:
                work_num_img = int(round(len_names * info_dir_data.proportion_of_dataset))

                if not self.silence_mode:
                    print(f"I take {work_num_img} of {len_names} imagess, proportion of dataset: {info_dir_data.proportion_of_dataset}")

                get_indexes = random.sample(indexes, work_num_img)
                img_names = [img_names_all[i] for i in get_indexes]

            else:
                get_indexes = indexes
                img_names = img_names_all

            self.list_img_name += img_names

            if not self.silence_mode:
                print(f"Conevrt list to img")
            dataset_imgs = [np.array(open_img_zip[1][i]) for i in get_indexes]

            if self.transform_data.mode_mask == 'separated' or \
               self.transform_data.mode_mask == 'image':
                open_mask_zip = np.load(info_dir_data.dir_mask_name, allow_pickle=True)
                if not self.silence_mode:
                    print(f"Conevrt list to mask")
                if self.transform_data.mode_mask == 'separated':
                    dataset_masks_glob = [np.array(open_mask_zip[1][i][:,:,:self.num_classes]) for i in get_indexes]
                else:
                    dataset_masks_glob = [np.array(open_mask_zip[1][i]) for i in get_indexes]

            elif self.transform_data.mode_mask == 'no_mask':
                # заглушка
                dataset_masks_glob = []
                for index_name, name in enumerate(
                        tqdm(img_names, file=sys.stdout, desc='Create fake masks', disable=self.silence_mode)):
                    dataset_masks_glob.append(np.zeros((*dataset_imgs[index_name].shape[:2],1), np.float32))
            else:
                raise AttributeError(
                    'The "mode_mask" parameter should be set to "separated", "image" or "no_mask" otherwise not implemented.')
            imgs += dataset_imgs
            masks_glob += dataset_masks_glob

        return imgs, masks_glob

    def saveNpyData(self, path="data/np_data_train"):
        if len(self.list_img_name) == 0:
            raise Exception("None save")

        if not os.path.isdir(path):
            print(f"create dir:'{path}'")
            os.makedirs(path)

        save_img_path = os.path.join(path, "original")
        save_img_arr = np.array((self.list_img_name, self.all_imges), dtype=object)
        np.save(save_img_path + '.npy', save_img_arr)

        if self.transform_data.mode_mask == 'separated' or \
           self.transform_data.mode_mask == 'image':
            save_mask_path = os.path.join(path, "mask")
            save_mask_arr = np.array((self.list_img_name, self.all_masks), dtype=object)
            np.save(save_mask_path + '.npy', save_mask_arr)

    def getTrainDataLoaderPytorch(self, num_workers = 0):
        gen = AugmentImages(self.all_imges, self.all_masks, self.gen_train.indexes, self.aug_dict, self.transform_data, self.augment)
        smart_gen = DataLoader(gen,
                               batch_size=self.transform_data.batch_size,
                               num_workers=num_workers
                               )
        return smart_gen

    def get_device_tensor_from_list_of_numpy(self, list_data):
        new_list_data = []
        for img in list_data:
            new_list_data.append(torch.from_numpy(img).to(self.device).permute(2, 0, 1))
        return new_list_data

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
