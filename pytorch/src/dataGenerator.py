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

def to_mean_std_format_img(in_img):
    mean = np.mean(in_img)
    std_dev = np.std(in_img)

    if std_dev != 0:
        return (in_img - mean) / std_dev
    else:
        return in_img

class InfoDirData:
    def __init__(self,
                 common_dir_path=None,
                 dir_img_name=None,
                 dir_img_path="data/train/origin",
                 dir_mask_name="data/train/",
                 add_mask_prefix='mask_',
                 proportion_of_dataset = 1.0,
                 proportion_taking_type="random"):
        self.dir_img_name = dir_img_path if dir_img_name is None else dir_img_name
        self.dir_mask_name = dir_mask_name
        self.add_mask_prefix = add_mask_prefix
        self.proportion_of_dataset = proportion_of_dataset
        self.common_dir_path = common_dir_path
        self.proportion_taking_type = proportion_taking_type

    def __str__(self):
        return f"\nInfoDirData:"+ \
               f"\n\t common_dir_path: {self.common_dir_path}" if self.common_dir_path is not None else "" +\
               f"\n\t dir_img_name: {self.dir_img_name}" +\
               f"\n\t dir_mask_name: {self.dir_mask_name}" if self.common_dir_path is None else "" + \
               f"\n\t add_mask_prefix: {self.add_mask_prefix}" + \
               f"\n\t proportion_of_dataset: {self.proportion_of_dataset}" + \
               f"\n\t proportion_of_dataset: {self.proportion_taking_type}"

    def __repr__(self):
        return self.__str__()

class TransformData:
    def __init__(self,
                 color_mode_img='gray',
                 mode_mask='separated',
                 target_size=(256, 256),
                 batch_size=1,
                 binary_mask=True,
                 normalization_img_fun="dev 255",
                 normalization_mask_fun="dev 255"):
        self.color_mode_img = color_mode_img
        self.mode_mask = mode_mask
        self.target_size = target_size
        self.batch_size = batch_size
        self.binary_mask = binary_mask
        self.normalization_img_fun = normalization_img_fun
        self.normalization_mask_fun = normalization_mask_fun

    def __str__(self):
        return str(self.__dict__)

class SaveData:
    def __init__(self,
                 save_to_dir=None,
                 save_prefix_image="image_",
                 save_prefix_mask="mask_"):
        self.save_to_dir = save_to_dir
        self.save_prefix_image = save_prefix_image
        self.save_prefix_mask = save_prefix_mask

    def __str__(self):
        return str(self.__dict__)



class TailingData:
    def __init__(self,
                 tailing=None):
        self.tailing=tailing

    def __str__(self):
        return str(self.__dict__)

class AugmentGenerator(Dataset):
    def __init__(self, images, masks, transform_data, aug_dict, mode, save_inform, tailing, augment, device, num_gen_repetitions):
        self.transform_data = transform_data
        self.mode = mode
        self.save_inform = save_inform[0]
        self.inform_generator = save_inform[1]
        self.augment = augment
        self.batch_size = self.transform_data.batch_size
        self.device = device

        self.images = images
        self.masks = masks
        self.indexes = None
        self.num_gen_repetitions = num_gen_repetitions

        self.tailing=tailing
        self.tiling_fun = self.get_tiling_fun(tailing)

        if self.inform_generator == "validation generator":
            self.set_validation_setting()

        self.composition = create_transform(aug_dict, self.transform_data, self.augment)

    def __len__(self):
        'Denotes the number of batches per epoch'

        len_gen = int(np.floor(len(self.indexes)*(self.num_gen_repetitions+1)/ self.transform_data.batch_size))
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

            X, y = self.batch_transform(X, y)
            X = torch.stack(X)
            y = torch.stack(y)
            return X, y

        elif self.mode == 'predict' or self.mode == "test":
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

    def get_tiling_fun(self, tailing):
        if tailing == False:
            return lambda x,y: (x,y)
        else:
            return self.tiling_img

    def batch_transform(self, img_batch, masks_batch):
        for i in range(self.batch_size):
            img,mask = self.tiling_fun(img_batch[i], masks_batch[i])
            img_batch[i], masks_batch[i] = self.random_transform_one_frame(img,mask)
        return img_batch, masks_batch


    def tiling_img(self, x, y):
        shape_y, shape_x = x.shape[-2:]

        size_y, size_x = self.transform_data.target_size

        y_pos = np.random.randint(0, shape_y - size_y+1)
        x_pos = np.random.randint(0, shape_x - size_x+1)

        ret_x = x[..., y_pos:y_pos + size_y, x_pos:x_pos + size_x]
        ret_y = y[..., y_pos:y_pos + size_y, x_pos:x_pos + size_x]

        return ret_x, ret_y

    def set_validation_setting(self):
        if self.augment == True:
            print("\nINFO: I'm a validation generator turn off augmentation for myself !\n", flush=True)
            self.augment = False

        # def on_epoch_end(self):
    #    print()
    ##    print(self.inform_generator)
    ##    print(self.small_list_img_name)
    #    print()


class DataGeneratorReaderAll():
    'Generates data with reading all data in RAM'
    def __init__(self,
                 dir_data=InfoDirData(),
                 list_class_name=None,
                 num_classes=2,
                 mode='train',
                 tailing=False,
                 num_gen_repetitions=0,
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
                 device="cuda"):

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
        self.num_gen_repetitions = num_gen_repetitions

        self.silence_mode = silence_mode

        self.img_normalization_fun  = self.get_normalization_fun(self.transform_data.normalization_img_fun)
        self.mask_normalization_fun = self.get_normalization_fun(self.transform_data.normalization_mask_fun)

        self.list_img_name = []

        self.all_imges, self.all_masks = self.load_data(type_load_data)

        self.all_imges = self.get_device_tensor_from_list_of_numpy(self.all_imges)
        self.all_masks = self.get_device_tensor_from_list_of_numpy(self.all_masks)

        self.indexes = [i for i in range(len(self.list_img_name))]
        print("In normal dir:", len(self.list_img_name), "images", flush=True)

        np.random.seed(self.seed)

        generator_param_dic = {
            "images": self.all_imges,
            "masks": self.all_masks,
            "transform_data": transform_data,
            "aug_dict": aug_dict,
            "mode": mode,
            "tailing": tailing,
            "augment": augment,
            "device": device,
            "num_gen_repetitions": num_gen_repetitions
        }

        if self.mode == "train":
            self.gen_train = AugmentGenerator(**generator_param_dic,
                                              save_inform=[save_inform, "train generator"])
            self.gen_valid = AugmentGenerator(**generator_param_dic,
                                              save_inform=[save_inform, "validation generator"])
        else:
            self.gen_test = AugmentGenerator(**generator_param_dic,
                                             save_inform=[save_inform, "test generator"])

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'

        len_gen = int(np.floor(len(self.list_img_name)*(self.num_gen_repetitions+1)/self.transform_data.batch_size))
        if len_gen == 0:
            print("very small dataset, please add more img")

        # print(int(len(self.list_img_name) * self.share_val))
        # print(int(len(self.list_train_img_name) / self.transform_data.batch_size))
        # print(int(len(self.list_validation_img_name) / self.transform_data.batch_size))

        return len_gen
    def get_normalization_fun(self, name):
        if name == "dev 255":
            return to_0_1_format_img
        elif name == "mean std":
            self.img_normalization_fun = to_mean_std_format_img
        else:
            raise Exception(f"normalization function '{name}' is not define")

    def load_data(self, type_load_data):
        if type_load_data == "img":
            return self.loadImgData()
        elif type_load_data == "npy":
            return self.loadNpyData()
        else:
            raise Exception("Now you can choose between loading by images or using an .npy file")

    def load_one_img(self, img_path):
        if self.transform_data.color_mode_img == "rgb":
            image = io.imread(img_path)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif self.transform_data.color_mode_img == "hsv":
            img = io.imread(img_path)
            image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)
        elif self.transform_data.color_mode_img == "gray":
            img = io.imread(img_path, as_gray=True)
            image = np.expand_dims(img, axis=-1)
        else:
            raise Exception(f"Don't known color_mode_img '{self.transform_data.color_mode_img}' ")

        return image.astype(np.float32)

    def loadImgData(self):
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

            path_to_img_dir = info_dir_data.dir_img_name if info_dir_data.common_dir_path is None else\
                              os.path.join(info_dir_data.common_dir_path, info_dir_data.dir_img_name)

            if not self.silence_mode and len(list_InfoDirData) > 1:
                print(f"Load {i+1} dataset of {len(list_InfoDirData)}")
            if not os.path.exists(path_to_img_dir):
                print("Image path not found!")
                raise AttributeError(f"Image path '{path_to_img_dir}' not found!")

            img_names_all = [name for name in os.listdir(path_to_img_dir) if name.endswith((img_type))]
            len_names = len(img_names_all)

            if info_dir_data.proportion_of_dataset < 1:
                work_num_img = int(round(len_names * info_dir_data.proportion_of_dataset))
                if not self.silence_mode:
                    print(
                        f"I take {work_num_img} of {len_names} imagess, proportion of dataset: {info_dir_data.proportion_of_dataset}")

                if info_dir_data.proportion_taking_type=="random":
                    img_names = random.sample(img_names_all, work_num_img)
                elif info_dir_data.proportion_taking_type=="sequentially":
                    img_names = img_names[:work_num_img]
                else:
                    print("proportion_taking_type not found!")
                    raise AttributeError(f"proportion_taking_type '{info_dir_data.proportion_taking_type}' is unknown! proportion_taking_type may be only 'random' or 'sequentially' !")

            else:
                img_names = img_names_all

            self.list_img_name += img_names

            # Load images
            time.sleep(0.2)  # чтобы tqdm не печатал вперед print
            for name in tqdm(img_names, file=sys.stdout, desc='\tLoad slices', disable=self.silence_mode):
                img_path = os.path.join(path_to_img_dir, name)
                img = self.load_one_img(img_path)
                img = self.img_normalization_fun(img)
                # Store samples
                dataset_imgs.append(img)

            mask_dirs_path = info_dir_data.dir_mask_name if info_dir_data.common_dir_path is None else\
                             info_dir_data.common_dir_path

            # Load masks
            if self.transform_data.mode_mask == 'separated':
                for index_name, name in enumerate(tqdm(img_names, file=sys.stdout, desc='\tLoad masks', disable=self.silence_mode)):

                    class_mask = np.zeros((*dataset_imgs[index_name].shape[:2], self.num_classes), np.float32)

                    for i in range(self.num_classes):
                        mask_path = os.path.join(mask_dirs_path, self.list_class_name[i],
                                                 info_dir_data.add_mask_prefix + name)

                        if os.path.isfile(mask_path):
                            mask_read = io.imread(mask_path, as_gray=True)
                            masks = mask_read.astype(np.float32)
                            masks = self.mask_normalization_fun(masks)
                            if self.transform_data.binary_mask:
                                masks[masks<0.5]=0.0
                                masks[masks>0.4]=1.0
                        else:
                            print("no open ", mask_path)
                            masks = np.zeros(dataset_imgs[index_name].shape[:2], np.float32)

                        class_mask[:, :, i] = masks
                    dataset_masks_glob.append(class_mask)

            elif self.transform_data.mode_mask == 'image':
                for index_name, name in enumerate(
                        tqdm(img_names, file=sys.stdout, desc='\tLoad res image', disable=self.silence_mode)):
                    mask_path = os.path.join(mask_dirs_path, self.list_class_name[0],
                                            info_dir_data.add_mask_prefix + name)
                    if os.path.isfile(mask_path):
                        masks = self.load_one_img(mask_path)
                        masks = self.mask_normalization_fun(masks)
                    else:
                        print("no open ", mask_path)
                        raise Exception(f"ERROR! No open: {mask_path}")
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
            path_to_img_dir = info_dir_data.dir_img_name if info_dir_data.common_dir_path is None else \
                os.path.join(info_dir_data.common_dir_path, info_dir_data.dir_img_name)

            if not self.silence_mode and len(list_InfoDirData) > 1:
                print(f"Load {i+1} dataset of {len(list_InfoDirData)}")
            if not os.path.exists(path_to_img_dir):
                print("Image path not found!")
                raise AttributeError(f"Image path '{path_to_img_dir}' not found!")

            open_img_zip = np.load(path_to_img_dir, allow_pickle=True)

            img_names_all = open_img_zip[0]
            len_names = len(img_names_all)
            indexes = [i for i in range(len_names)]

            if info_dir_data.proportion_of_dataset < 1:
                work_num_img = int(round(len_names * info_dir_data.proportion_of_dataset))

                if not self.silence_mode:
                    print(f"I take {work_num_img} of {len_names} imagess, proportion of dataset: {info_dir_data.proportion_of_dataset}")

                if info_dir_data.proportion_taking_type=="random":
                    get_indexes = random.sample(indexes, work_num_img)
                    img_names = [img_names_all[i] for i in get_indexes]
                elif info_dir_data.proportion_taking_type=="sequentially":
                    get_indexes = indexes[:work_num_img]
                    img_names = img_names_all[:work_num_img]
                else:
                    print("proportion_taking_type not found!")
                    raise AttributeError(f"proportion_taking_type '{info_dir_data.proportion_taking_type}' is unknown! proportion_taking_type may be only 'random' or 'sequentially' !")



            else:
                get_indexes = indexes
                img_names = img_names_all

            self.list_img_name += img_names

            if not self.silence_mode:
                print(f"Conevrt list to img")
            dataset_imgs = [np.array(open_img_zip[1][i]) for i in get_indexes]

            # Read mask

            mask_dirs_path = info_dir_data.dir_mask_name if info_dir_data.common_dir_path is None else\
                               info_dir_data.common_dir_path
            if self.transform_data.mode_mask == 'separated' or \
               self.transform_data.mode_mask == 'image':
                open_mask_zip = np.load(mask_dirs_path, allow_pickle=True)
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
