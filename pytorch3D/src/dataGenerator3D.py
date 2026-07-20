import cv2
import numpy as np
import os
import random
import sys
import time
import torch

from torchvision import tv_tensors
from tqdm import tqdm
from torch.utils.data import Dataset

from augmentation3D import create_transform

from reader import get_img_loader, read_img, AVAILABLE_IMG_TYPE, read_nii, nii_type
from read_data_transform_fun import get_normalization_fun, enumerate_slice_transform
from text_fun import search_data_by_type, add_mods_in_file_name, get_part_of_list, name_list_sequence_checking

from DataStructures3D import InfoDirData, CommonTransformData, SaveGeneratorData, check_dataset_and_comon_transform_info#, TailingData,

from prepare_data import saveDataframeAsImgs
from read_data_transform_fun import to_0_255_format_img

class AugmentGenerator3D(Dataset):
    def __init__(self,
                 images,
                 masks,
                 transform_data,
                 aug_dict,
                 mode,
                 save_inform,
                 bool_tiling_list,
                 is_augment,
                 device,
                 class_statistic):
        self.transform_data = transform_data
        self.mode = mode
        self.save_inform = save_inform[0]
        self.inform_generator = save_inform[1]
        self.is_augment = is_augment
        self.device = device

        self.images = images
        self.masks = masks
        self.indexes = None

        self.bool_tiling_list=bool_tiling_list

        if self.inform_generator == "validation generator":
            self.set_validation_setting()

        self.composition = create_transform(aug_dict, self.transform_data, self.is_augment)
        self.class_statistic = class_statistic

    def __len__(self):
        'Denotes the number of batches per epoch'
        len_gen = len(self.indexes)//self.transform_data.batch_size + (0 if len(self.indexes)%self.transform_data.batch_size==0 else 1)
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
            if (index * self.transform_data.batch_size + i) < len(self.indexes):
                work_index = self.indexes[index * self.transform_data.batch_size + i]
                work_indexes.append(work_index)

        X = []
        bool_tiling_batch = []
        for work_index in work_indexes:
            X.append(self.images[work_index])
            bool_tiling_batch.append(self.bool_tiling_list[work_index])

        if self.mode == 'train':
            y = []
            for work_index in work_indexes:
                y.append(self.masks[work_index])

            X, y = self.batch_transform(X, y, bool_crop_batch=bool_tiling_batch)

            batch_statistic = None if self.class_statistic is None else self.calculate_batch_class_statictic(y)

            X = torch.stack(X)
            y = torch.stack(y)
            return X, y, batch_statistic

        elif self.mode == 'predict' or self.mode == "test":
            X = self.batch_test_transform(X)    # for resize
            X = torch.stack(X)
            return X
        else:
            raise AttributeError('The mode parameter should be set to "train" or "predict".')

    def calculate_batch_class_statictic(self, list_batch_data):
        num_classes = list_batch_data[0].shape[0]
        list_statistic = [0 for i in range(num_classes)]

        max_tiles_statistics = self.class_statistic["max_class_pixels_256"]
        '''
        str_count = "\n"
        for j, mask in enumerate(list_batch_data):
            str_count += f"\t{j} mask:"
            for i in range(num_classes):
                proportion_of_maximum = mask[i, :, :].sum() / max_tiles_statistics[i]
                if proportion_of_maximum > 0.1:
                    list_statistic[i] += proportion_of_maximum
                #else:
                #    list_statistic[i] += 1
                str_count += f"count {i} class = {mask[i, :, :].sum()}, "
            str_count+="\n"
        print("batch_stat_for_masks", str_count)
        '''
        for mask in list_batch_data:
            for i in range(num_classes):
                proportion_of_maximum = mask[i, :, :].sum() / max_tiles_statistics[i]
                if proportion_of_maximum > 0.1:
                    list_statistic[i] += proportion_of_maximum
                #else:
                #    list_statistic[i] += 1

        return [elem/self.transform_data.batch_size for elem in list_statistic]

        def random_transform_one_frame(self, img, masks):
            aug_img, aug_masks = self.composition(tv_tensors.Image(img),
                                                  tv_tensors.Mask(masks))
            if self.transform_data.binary_mask:
                aug_masks = aug_masks.round()
            return aug_img, aug_masks

    def batch_transform(self, img_batch, masks_batch, bool_crop_batch):
        for i in range(len(img_batch)):
            if bool_crop_batch[i] == True:
                img,mask = self.random_crop_img(img_batch[i], masks_batch[i])
            else:
                img, mask = img_batch[i], masks_batch[i]

            img_batch[i], masks_batch[i] = self.random_transform_one_frame(img,mask)
        return img_batch, masks_batch

    def batch_test_transform(self, img_batch):
        ret_batch = []
        for img in img_batch:
            ret_batch.append(self.composition(img))
        return ret_batch

    def random_crop_img(self, x, y):
        shape_y, shape_x = x.shape[-2:]

        size_y, size_x = self.transform_data.target_size

        y_pos = np.random.randint(0, shape_y - size_y+1)
        x_pos = np.random.randint(0, shape_x - size_x+1)

        ret_x = x[..., y_pos:y_pos + size_y, x_pos:x_pos + size_x]
        ret_y = y[..., y_pos:y_pos + size_y, x_pos:x_pos + size_x]

        return ret_x, ret_y

    def set_validation_setting(self):
        if self.is_augment == True:
            print("\nINFO: I'm a validation generator turn off augmentation for myself !\n", flush=True)
            self.is_augment = False

        # def on_epoch_end(self):
    #    print()
    ##    print(self.inform_generator)
    ##    print(self.small_list_img_name)
    #    print()

    def resize_mask_arr(self):
        for i in range(len(self.masks)):
            self.masks[i] = self.composition(self.masks[i])
        return self.masks

class DataGeneratorReaderAll3D:
    'Generates data with reading all data in RAM'
    def __init__(self,
                 dir_data=InfoDirData(),
                 list_class_name=None,
                 num_classes=2,
                 mode='train',
                 aug_dict=dict(),
                 is_augment=False,
                 is_shuffle=True,
                 #seed=42,
                 subsampling="crossover",
                 transform_data=CommonTransformData(),
                 save_inform=SaveGeneratorData(),
                 share_validat=0.2,
                 is_silence_mode=False,
                 is_calculate_statistic=False,
                 device="cuda",
                 debug_mode=False):

        self.typeGen = "DataGeneratorReaderAll3D"

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
        self.mode = mode
        self.save_inform = save_inform
        self.share_val = share_validat
        self.is_augment = is_augment
        self.aug_dict = aug_dict
        self.is_shuffle = is_shuffle
        self.subsampling = subsampling
        #self.seed = seed

        self.silence_mode = is_silence_mode

        #self.img_normalization_fun = self.get_normalization_fun(self.transform_data.normalization_img_fun)
        #self.mask_normalization_fun = self.get_normalization_fun(self.transform_data.normalization_mask_fun)

        self.all_dataframes,\
        self.all_masks,\
        self.indexes,\
        self.bool_tiling_list,\
        self.data_name_list,\
        self.repiter_list = self.loadData()

        if self.save_inform.save_to_dir is not None:
            save_path = self.save_inform.save_to_dir
            if not os.path.isdir(save_path):
                print("создаю out_dir:" + save_path.replace(u"\\\\?\\" + os.getcwd() + "\\", ""))
                os.makedirs(os.path.join(save_path))

            for i in range(len(self.data_name_list)):
                data_name = self.data_name_list[i]
                save_dir_path = os.path.join(save_path, data_name, "original")
                if not os.path.isdir(save_dir_path):
                    print("создаю imgs out_dir:" + os.path.join(save_dir_path).replace(u"\\\\?\\" + os.getcwd() + "\\", ""))
                    os.makedirs(os.path.join(save_dir_path))

                dataframe = self.all_dataframes[i]
                saveDataframeAsImgs(os.path.join(save_path, data_name, "original"),
                                    os.path.basename(data_name),
                                    to_0_255_format_img(dataframe))

                mask = self.all_masks[i]
                for c, class_name in enumerate(self.list_class_name):
                    save_maskdir_path = os.path.join(save_path, data_name, class_name)
                    if not os.path.isdir(save_maskdir_path):
                        print("создаю imgs out_dir:" + os.path.join(save_maskdir_path).replace(
                            u"\\\\?\\" + os.getcwd() + "\\", ""))
                    os.makedirs(os.path.join(save_maskdir_path))

                    saveDataframeAsImgs(save_maskdir_path,
                                        os.path.basename(data_name),
                                        to_0_255_format_img(mask[:,:,:,c]))

        self.all_dataframes = self.get_device_tensor_from_list_of_numpy(self.all_dataframes)
        self.all_masks      = self.get_device_tensor_from_list_of_numpy(self.all_masks)

        #if is_calculate_statistic:
        #    self.calculate_tensor_dataset_classes_statistic()
        #else:
        self.class_statistic = None

        print(f"In normal dir:{len(self.indexes)} indexes, and {len(self.all_dataframes)} images", flush=True)

        #np.random.seed(self.seed)

        generator_param_dic = {
            "dataframes": self.all_dataframes,
            "masks": self.all_masks,
            "transform_data": transform_data,
            "aug_dict": aug_dict,
            "mode": mode,
            "bool_tiling_list": self.bool_tiling_list,
            "is_augment": is_augment,
            "device": device,
            "class_statistic": self.class_statistic
        }

        if self.mode == "train":
            self.gen_train = AugmentGenerator3D(**generator_param_dic,
                                                save_inform=[save_inform, "train generator"])
            self.gen_valid = AugmentGenerator3D(**generator_param_dic,
                                                save_inform=[save_inform, "validation generator"])
        else:
            self.gen_test = AugmentGenerator3D(**generator_param_dic,
                                               save_inform=[save_inform, "test generator"])

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        len_gen = int(np.floor(len(self.indexes)/self.transform_data.batch_size))
        if len_gen == 0:
            print("very small dataset, please add more img")

        # print(int(len(self.list_img_name) * self.share_val))
        # print(int(len(self.list_train_img_name) / self.transform_data.batch_size))
        # print(int(len(self.list_validation_img_name) / self.transform_data.batch_size))

        return len_gen

    def get_dataset_info(self):
        return {
            "type gen": self.typeGen,
            "class names": self.list_class_name,
            "number of classes": self.num_classes,
            "dir config": self.dir_data,
            "transform config": self.transform_data,
            "augmentation dict": self.aug_dict,
            "type prepare data (mode)": self.mode,
            "subsampling": self.subsampling,
            "proportion of validation data (of all)": self.share_val,
            "is augmentation": self.is_augment,
            "is shuffle": self.is_shuffle,
            "save data information": self.save_inform,
            #"random seed": self.seed,
            "number of images": len(self.data_name_list),
            "number of tiles": len(self.indexes),
            "dataset statistic": self.class_statistic,
            "is silence_mode": self.silence_mode,
            "device": self.device
        }

    '''
    @staticmethod
    def count_max_window_value_on_img (mask, window_size):
        h,w = mask.shape
        w_h, w_w = window_size

        #print(max([mask[y:y+w_h, x:x+w_w].sum() for x in range(w-w_w+1) for y in range (h-w_h+1)]))
        max_value = 0
        for y in range(h - w_h + 1):
            work_value = mask[y:y+w_h, 0:w_w].sum()
            str_value = [work_value]
            for x in range(1, w - w_w + 1):
                work_value += -mask[y:y+w_h, x-1:x].sum()+mask[y:y+w_h, x+w_w-1:x+w_w].sum()
                str_value.append(work_value)
            max_value = max(max_value, max(str_value))

        return max_value

    # подсчет статистик обычного датасета (не тензора!)
    def calculate_img_dataset_classes_statistic(self):
        num_pixels = 0
        list_num_class_pixels = [0 for i in range(self.num_classes)]
        list_max_class_pixels = [0 for i in range(self.num_classes)]

        for mask in tqdm(self.all_masks, file=sys.stdout, desc='\tCalculate statistic', disable=self.silence_mode):
            num_pixels+= mask[:,:,0].size
            #print(num_pixels, mask.shape)

            for class_index in range(self.num_classes):
                list_num_class_pixels[class_index] += mask[:, :, class_index].sum()
                #print("\t", list_num_class_pixels[class_index])

                max_title_from_image = self.count_max_window_value_on_img(mask[:, :, class_index], self.transform_data.target_size)

                if list_max_class_pixels[class_index] < max_title_from_image:
                    list_max_class_pixels[class_index] = max_title_from_image

        self.class_statistic = {"p_classes": [], "max_class_pixels_256": list_max_class_pixels}
        for class_index in range(self.num_classes):
            self.class_statistic["p_classes"].append(list_num_class_pixels[class_index]/num_pixels)

        if not self.silence_mode:
            print("\nnumpy class_statistic", self.class_statistic)
        return self.class_statistic

    def calculate_tensor_dataset_classes_statistic(self):
        num_pixels = 0
        list_num_class_pixels = [0 for i in range(self.num_classes)]
        list_max_class_pixels = [0 for i in range(self.num_classes)]

        for mask in tqdm(self.all_masks, file=sys.stdout, desc='\tCalculate statistic', disable=self.silence_mode):
            num_pixels+= mask[0, :,:].size().numel()
            #print(num_pixels, mask.shape)
            #print(mask.get_device())

            conv = torch.nn.Conv2d(in_channels=1,
                                   out_channels=1,
                                   kernel_size=self.transform_data.target_size,
                                   device=mask.get_device()
                                   )

            kernel = torch.ones(self.transform_data.target_size).to(mask.get_device())

            kernel_tensor = torch.unsqueeze(torch.unsqueeze(kernel, 0), 0)  # size: (1, 1, k, k)
            conv.weight = torch.nn.Parameter(kernel_tensor)

            for class_index in range(self.num_classes):
                list_num_class_pixels[class_index] += mask[class_index, :, :].sum()
                #print("\t", list_num_class_pixels[class_index])

                mask2conv = conv(torch.unsqueeze(torch.unsqueeze(mask[class_index, :, :], 0), 0))

                max_title_from_image = mask2conv.max().type(torch.int64)

                if list_max_class_pixels[class_index] < max_title_from_image:
                    list_max_class_pixels[class_index] = max_title_from_image

        self.class_statistic = {"p_classes": [], "max_class_pixels_256": torch.stack(list_max_class_pixels, -1).detach().cpu().numpy()}
        for class_index in range(self.num_classes):
            self.class_statistic["p_classes"].append(list_num_class_pixels[class_index]/num_pixels)
        self.class_statistic["p_classes"] = torch.stack(self.class_statistic["p_classes"], -1).detach().cpu().numpy()

        if not self.silence_mode:
            print("\ntorch class_statistic", self.class_statistic)
        return self.class_statistic
    '''


    def load_one_image_dataset(self, info_dir_data:InfoDirData, index_overlap:int=0):
        imgs = []
        masks_glob = []

        repiter_counter = info_dir_data.num_gen_repetitions + 1
        list_indexes = [index_overlap for i in range(repiter_counter)]

        path_to_img_dir = info_dir_data.dir_img_name if info_dir_data.common_dir_path is None else \
            os.path.join(info_dir_data.common_dir_path, info_dir_data.dir_img_name)

        # search files
        if not os.path.exists(path_to_img_dir):
            print("Image path not found!")
            raise AttributeError(f"Image path '{path_to_img_dir}' not found!")

        img_names_all = [name for name in os.listdir(path_to_img_dir) if name.endswith((AVAILABLE_IMG_TYPE))]

        #  sorting and check
        sorted_data_name_sequence = name_list_sequence_checking(img_names_all, self.transform_data.target_size[0])
        if sorted_data_name_sequence is None:
            raise RuntimeError(f'The data in "{path_to_img_dir}" cannot be stacked.')
        else:
            list_img_name = sorted_data_name_sequence

        if len(list_img_name) == 0:
            print("WARNING!!! Dataset is clear!!!")
        else:
            data_reader = get_img_loader(check_dataset_and_comon_transform_info(info_dir_data,
                                                                                self.transform_data,
                                                                                "type_load_data",
                                                                                check_none=True))

            data_normalizer = get_normalization_fun(check_dataset_and_comon_transform_info(info_dir_data,
                                                                                self.transform_data,
                                                                                "normalization_data_fun"))

            use_mask_loader = check_dataset_and_comon_transform_info(info_dir_data,
                                                                     self.transform_data,
                                                                     "type_load_target",
                                                                     check_none=True)

            mask_normalizer = get_normalization_fun(check_dataset_and_comon_transform_info(info_dir_data,
                                                                                self.transform_data,
                                                                                "normalization_mask_fun"))

            use_binary_mask = check_dataset_and_comon_transform_info(info_dir_data,
                                                                    self.transform_data,
                                                                    "binary_mask")

            # Load images
            try:
                cmd_size = os.get_terminal_size().columns
            except Exception:
                cmd_size = 200
            ncols = cmd_size - 8
            time.sleep(0.2)  # чтобы tqdm не печатал вперед print
            for name in tqdm(list_img_name, ncols=ncols, file=sys.stdout, desc='\tLoad slices', disable=self.silence_mode):
                img_path = os.path.join(path_to_img_dir, name)
                img = data_reader(img_path)
                img = data_normalizer(img)
                # Store samples
                imgs.append(img)
                repiter_list.append(info_dir_data.num_gen_repetitions + 1)
                bool_tailing_list.append(info_dir_data.cropping_data)
                list_indexes += [counter_index for i in range(info_dir_data.num_gen_repetitions + 1)]
                counter_index += 1

            # Load masks
            mask_dirs_path = info_dir_data.dir_mask_name if info_dir_data.common_dir_path is None else \
                info_dir_data.common_dir_path

            if use_mask_loader == 'separated':
                for index_name, name in enumerate(
                        tqdm(list_img_name, file=sys.stdout, ncols=ncols,desc='\tLoad masks', disable=self.silence_mode)):

                    class_mask = np.zeros((*imgs[index_name].shape[:2], self.num_classes), np.float32)

                    for i in range(self.num_classes):
                        mask_path = os.path.join(mask_dirs_path,
                                                 self.list_class_name[i],
                                                 add_mods_in_file_name(info_dir_data.add_mask_prefix,
                                                                       name,
                                                                       info_dir_data.add_mask_suffix)
                                                 )

                        if os.path.isfile(mask_path):
                            mask_read = read_img(mask_path, as_gray=True)
                            masks = mask_read.astype(np.float32)
                            masks = mask_normalizer(masks)
                            if use_binary_mask:
                                masks[masks < 0.5] = 0.0
                                masks[masks > 0.4] = 1.0
                            #masks.resize(masks.shape[:2])
                        else:
                            print("no open ", mask_path)
                            masks = np.zeros(imgs[index_name].shape[:2], np.float32)

                        class_mask[:, :, i] = masks
                    masks_glob.append(class_mask)

            elif use_mask_loader == 'image':
                for index_name, name in enumerate(
                        tqdm(list_img_name, file=sys.stdout, ncols=ncols, desc='\tLoad res image', disable=self.silence_mode)):
                    mask_path = os.path.join(mask_dirs_path,
                                             self.list_class_name[0],
                                             add_mods_in_file_name(info_dir_data.add_mask_prefix,
                                                                   name,
                                                                   info_dir_data.add_mask_suffix)
                                             )

                    if os.path.isfile(mask_path):
                        masks = data_reader(mask_path)
                        masks = data_normalizer(masks)
                    else:
                        print("no open ", mask_path)
                        raise Exception(f"ERROR! No open: {mask_path}")
                    # print(masks.shape)
                    masks_glob.append(masks)

            elif use_mask_loader == 'no_mask':
                # заглушка
                for index_name, name in enumerate(
                        tqdm(list_img_name, file=sys.stdout, ncols=ncols, desc='Create fake masks', disable=self.silence_mode)):
                    masks_glob.append(np.zeros((*imgs[index_name].shape[:2], 1), np.float32))
            else:
                raise AttributeError(
                    'The "type_load_target" parameter should be set to "separated", "image" or "no_mask" otherwise not implemented.')

        return imgs, masks_glob, list_indexes, bool_tailing_list, list_img_name, repiter_list

################################################################################################################################ check and fixed
    def load_one_numpy_dataset(self, info_dir_data, index_overlap=0):
        list_img_name = []
        list_indexes = []
        counter_index = index_overlap

        path_to_img_dir = info_dir_data.dir_img_name if info_dir_data.common_dir_path is None else \
            os.path.join(info_dir_data.common_dir_path, info_dir_data.dir_img_name)

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
                print(
                    f"I take {work_num_img} of {len_names} imagess, proportion of dataset: {info_dir_data.proportion_of_dataset}")

            if info_dir_data.proportion_taking_type == "random":
                get_indexes = random.sample(indexes, work_num_img)
                img_names = [img_names_all[i] for i in get_indexes]
            elif info_dir_data.proportion_taking_type == "sequentially":
                get_indexes = indexes[:work_num_img]
                img_names = img_names_all[:work_num_img]
            else:
                print("proportion_taking_type not found!")
                raise AttributeError(
                    f"proportion_taking_type '{info_dir_data.proportion_taking_type}' is unknown! proportion_taking_type may be only 'random' or 'sequentially' !")

        else:
            get_indexes = indexes
            img_names = img_names_all

        list_img_name += img_names

        if not self.silence_mode:
            print(f"Conevrt list to img")
        imgs = [np.array(open_img_zip[1][i]) for i in get_indexes]
        repiter_list = [np.array(open_img_zip[2][i]) for i in get_indexes]
        bool_tailing_list = [np.array(open_img_zip[3][i]) for i in get_indexes]

        for i in get_indexes:
            list_indexes += [counter_index for i in range(np.array(open_img_zip[2][i]) + 1)]
            counter_index += 1

        # Read mask
        mask_dirs_path = info_dir_data.dir_mask_name if info_dir_data.common_dir_path is None else \
            info_dir_data.common_dir_path
        if self.transform_data.mode_mask == 'separated' or \
                self.transform_data.mode_mask == 'image':
            open_mask_zip = np.load(mask_dirs_path, allow_pickle=True)
            if not self.silence_mode:
                print(f"Conevrt list to mask")
            if self.transform_data.mode_mask == 'separated':
                masks_glob = [np.array(open_mask_zip[1][i][:, :, :self.num_classes]) for i in
                                      get_indexes]
            else:
                masks_glob = [np.array(open_mask_zip[1][i]) for i in get_indexes]

        elif self.transform_data.mode_mask == 'no_mask':
            masks_glob = []
            # заглушка
            for index_name, name in enumerate(
                    tqdm(img_names, file=sys.stdout, desc='Create fake masks', disable=self.silence_mode)):
                masks_glob.append(np.zeros((*imgs[index_name].shape[:2], 1), np.float32))
        else:
            raise AttributeError(
                'The "mode_mask" parameter should be set to "separated", "image" or "no_mask" otherwise not implemented.')

        return imgs, masks_glob, list_indexes, bool_tailing_list, list_img_name, repiter_list

    def load_one_nii_dataset(self, info_dir_data, index_overlap=0):
        found_type = nii_type

        imgs = []
        masks_glob = []
        repiter_list = []

        list_indexes = []
        bool_tailing_list = []
        counter_index = index_overlap

        list_img_name_all = []

        path_to_img_dir = info_dir_data.dir_img_name if info_dir_data.common_dir_path is None else \
            os.path.join(info_dir_data.common_dir_path, info_dir_data.dir_img_name)

        # search files
        if not os.path.exists(path_to_img_dir):
            print("Image path not found!")
            raise AttributeError(f"Image path '{path_to_img_dir}' not found!")
        img_names_all = search_data_by_type(path_to_img_dir, found_type, info_dir_data.ignore_data_suffix)
        #print(img_names_all)
        len_path = len(path_to_img_dir)
        #print(len_path)
        img_names_all = [name[len_path:] for name in img_names_all]
        #print(img_names_all)

        #  select the amount of data
        list_img_name = get_part_of_list(img_names_all,
                                         info_dir_data.proportion_of_dataset,
                                         info_dir_data.proportion_taking_type)

        if len(list_img_name) == 0:
            print("WARNING!!! Dataset is clear!!!")
        else:
            data_normalizer = get_normalization_fun(check_dataset_and_comon_transform_info(info_dir_data,
                                                                                self.transform_data,
                                                                                "normalization_data_fun"))

            use_mask_loader = check_dataset_and_comon_transform_info(info_dir_data,
                                                                     self.transform_data,
                                                                     "type_load_target",
                                                                     check_none=True)

            mask_normalizer = get_normalization_fun(check_dataset_and_comon_transform_info(info_dir_data,
                                                                                self.transform_data,
                                                                                "normalization_mask_fun"))

            #####################################################################################################################3####
            max_shape = [0, 0]
            min_shape = [500, 550]
            shapes_h = []
            shapes_w = []
            num_val = 0
            square = []

            max_val = 0
            min_val = 9000

            max_mask = 0
            min_mask = 9000

            n_slices_list = []
            #############################################################################################################################

            try:
                cmd_size = os.get_terminal_size().columns
            except Exception:
                cmd_size = 200
            ncols = cmd_size - 8
            # Load images
            time.sleep(0.2)  # чтобы tqdm не печатал вперед print
            for name in tqdm(list_img_name, file=sys.stdout, ncols=ncols, desc='\tLoad slices', disable=self.silence_mode):
                img_path = os.path.join(path_to_img_dir, name)
                images = read_nii(img_path)
                images = data_normalizer(images)

                #######################################################################################################################
                hei, wei = images.shape[0:2]
                if hei < min_shape[0]:
                    min_shape[0] = hei
                elif hei > max_shape[0]:
                    max_shape[0] = hei

                if wei < min_shape[1]:
                    min_shape[1] = wei
                elif wei > max_shape[1]:
                    max_shape[1] = wei

                shapes_h.append(hei)
                shapes_w.append(wei)
                num_val += 1
                square.append(wei * hei)

                max_img = images.max()
                min_img = images.min()

                if max_img > max_val:
                    max_val = max_img

                if min_img < min_val:
                    min_val = min_img

                n_slices_list.append(images.shape[2])
                ####################################################################################################################

                for slice_index in range(images.shape[2]):
                    img = images[:, :, slice_index]
                    img = np.expand_dims(img, -1)
                    imgs.append(img)

                    repiter_list.append(info_dir_data.num_gen_repetitions + 1)
                    bool_tailing_list.append(info_dir_data.cropping_data)
                    list_indexes += [counter_index for i in range(info_dir_data.num_gen_repetitions + 1)]
                    counter_index += 1

                    list_img_name_all.append(add_mods_in_file_name("", name, f"_{slice_index}"))

            mask_dirs_path = info_dir_data.dir_mask_name if info_dir_data.common_dir_path is None else \
                info_dir_data.common_dir_path

            # Load masks
            if use_mask_loader == "nii":
                for index_name, name in enumerate(
                        tqdm(list_img_name, file=sys.stdout, ncols=ncols, desc='\tLoad masks', disable=self.silence_mode)):
                    mask_path = os.path.join(mask_dirs_path,
                                             #self.list_class_name[0],
                                             add_mods_in_file_name(info_dir_data.add_mask_prefix,
                                                                   name,
                                                                   info_dir_data.add_mask_suffix)
                                             )
                    if os.path.isfile(mask_path):
                        masks = read_nii(mask_path)
                        for slice_index in range(masks.shape[2]):
                            class_mask = enumerate_slice_transform(masks[:, :, slice_index])

                            for class_index in range(class_mask.shape[2]):
                                class_mask[:,:, class_index] = mask_normalizer(class_mask[:,:, class_index])

                            ########################################################################################################################
                            max_mask_val = class_mask.max()
                            min_mask_val = class_mask.min()

                            if max_mask_val > max_mask:
                                max_mask = max_mask_val

                            if min_mask_val < min_mask:
                                min_mask = min_mask_val
                            ##########################################################################################################################

                            if class_mask.shape[2] < self.num_classes:
                                axis = -1
                                pad_size = self.num_classes - class_mask.shape[axis]
                                npad = [(0, 0)] * class_mask.ndim
                                npad[axis] = (0, pad_size)
                                class_mask = np.pad(class_mask, pad_width=npad, mode='constant', constant_values=0)
                            elif class_mask.shape[2] > self.num_classes:  # обрезка лишних
                                class_mask = class_mask[:, :, :self.num_classes]

                            masks_glob.append(class_mask)
                    else:
                        print("no open ", mask_path)
                        raise Exception(f"ERROR! No open: {mask_path}")
                    # print(masks.shape)
                    # masks_glob.append(masks)

            else:
                raise AttributeError(
                    'The "mode_mask" parameter should be set to "nii" otherwise not implemented.')

            print(max_shape, min_shape)
            print("square mean", sum(square) / num_val, "square min", min(square), "square max", max(square))

            print("shapes h mean", sum(shapes_h) / num_val, "shapes h min", min(shapes_h), "shapes h max",
                  max(shapes_h))
            print("shapes w mean", sum(shapes_w) / num_val, "shapes w min", min(shapes_w), "shapes w max",
                  max(shapes_w))

            print("min max val ", min_val, max_val)

            print("mask min max val ", min_mask, max_mask)

            print("min_z", min(n_slices_list), "max_z", max(n_slices_list), ", mean_z", sum(n_slices_list)/num_val)

        return imgs, masks_glob, list_indexes, bool_tailing_list, list_img_name_all, repiter_list

    def loadData(self):
        imgs = []
        masks_glob = []
        repiter_list = []
        bool_tailing_list = []
        list_indexes = []
        list_img_name = []
        counter_index = 0

        if isinstance(self.dir_data, InfoDirData):
            list_InfoDirData = [self.dir_data]
        elif all(isinstance(x, InfoDirData) for x in self.dir_data): # for iteraitable type such as list or turple
            list_InfoDirData = self.dir_data
        else:
            raise Exception(f"ERROR type dir_data don't know: {self.dir_data}")

        for i, info_dir_data in enumerate(list_InfoDirData):
            if len(list_InfoDirData) > 1: # and not self.silence_mode:
                print(f"Load {i + 1} dataset of {len(list_InfoDirData)}")

            use_load_type_dataset = check_dataset_and_comon_transform_info(info_dir_data,
                                                                           self.transform_data,
                                                                           "type_load_data",
                                                                           check_none=True)

            if use_load_type_dataset in ["rgb", "hsv", "gray"]:
                 dataset_imgs,\
                 dataset_masks_glob,\
                 dataset_list_indexes,\
                 dataset_bool_tailing_list,\
                 dataset_list_img_name,\
                 dataset_repiter_list = self.load_one_image_dataset(info_dir_data, counter_index)
            elif use_load_type_dataset == "npy":
                 dataset_imgs,\
                 dataset_masks_glob,\
                 dataset_list_indexes,\
                 dataset_bool_tailing_list,\
                 dataset_list_img_name,\
                 dataset_repiter_list = self.load_one_numpy_dataset(info_dir_data, counter_index)
            elif use_load_type_dataset in ["nii", "nii.gz"]:
                 dataset_imgs, \
                 dataset_masks_glob, \
                 dataset_list_indexes, \
                 dataset_bool_tailing_list, \
                 dataset_list_img_name, \
                 dataset_repiter_list = self.load_one_nii_dataset(info_dir_data, counter_index)
            else:
                raise Exception(f'ERROR {i} dataset with type "{use_load_type_dataset}". Now you can choose between loading by images ("rgb", "hsv", "gray"), nii ("nii", "nii.gz"), and npy ("npy") files')

            imgs += dataset_imgs
            masks_glob += dataset_masks_glob
            list_indexes += dataset_list_indexes
            bool_tailing_list += dataset_bool_tailing_list
            list_img_name += dataset_list_img_name
            repiter_list += dataset_repiter_list
            counter_index += len(dataset_imgs)

        return imgs, masks_glob, list_indexes, bool_tailing_list, list_img_name, repiter_list

    def saveNpyData(self, path="data/np_data_train"):
        if len(self.data_name_list) == 0:
            raise Exception("None save")

        if not os.path.isdir(path):
            print(f"create dir:'{path}'")
            os.makedirs(path)

        all_imges = self.get_cpu_numpy_from_list_of_tensor(self.all_imges)
        all_masks = self.get_cpu_numpy_from_list_of_tensor(self.all_masks)

        save_img_path = os.path.join(path, "original")
        save_img_arr = np.array((self.data_name_list,
                                 all_imges,
                                 self.repiter_list,
                                 self.bool_tiling_list), dtype=object)
        np.save(save_img_path + '.npy', save_img_arr)

        if self.transform_data.mode_mask == 'separated' or \
                self.transform_data.mode_mask == 'image':
            save_mask_path = os.path.join(path, "mask")
            save_mask_arr = np.array((self.data_name_list, all_masks), dtype=object)
            np.save(save_mask_path + '.npy', save_mask_arr)

    def get_device_tensor_from_list_of_numpy(self, list_data):
        new_list_data = []
        for img in list_data:
            new_list_data.append(torch.from_numpy(img).to(self.device).permute(2, 0, 1))
        return new_list_data

    def get_cpu_numpy_from_list_of_tensor(self, list_data):
        new_list_data = []
        for img in list_data:
            new_list_data.append(img.detach().cpu().permute(1, 2, 0).numpy())
        return new_list_data

    def on_epoch_end(self):
        len_generator_titlels = len(self.indexes)
        size_val = int(round(len_generator_titlels * self.share_val))

        'Updates indexes after each epoch'
        # print("change generator")

        if self.mode == 'train':
            self.gen_train.indexes = self.indexes[:len_generator_titlels - size_val]
            self.gen_valid.indexes = self.indexes[len_generator_titlels - size_val:]

            # print(self.indexes[:len(self.list_img_name) - size_val])
            # print(self.indexes[len(self.list_img_name) - size_val:])

        elif self.mode in ['predict', "test"]:
            self.gen_test.indexes = self.indexes
        else:
            raise AttributeError('The "mode" parameter should be set to "train" or "predict".')

        if self.is_shuffle is True:
            print("shuffling in the generator")
            # print()

            if self.subsampling == 'random':
                list_shuffle_indexes_train = self.indexes[:len_generator_titlels - size_val]
                list_shuffle_indexes_test = self.indexes[len_generator_titlels - size_val:]
                np.random.shuffle(list_shuffle_indexes_train)
                np.random.shuffle(list_shuffle_indexes_test)

                self.indexes = list_shuffle_indexes_test + list_shuffle_indexes_train
                #print(self.indexes, list_shuffle_indexes_train, list_shuffle_indexes_test)

            elif self.subsampling == 'crossover':
                self.indexes = self.indexes[len_generator_titlels - size_val:] + self.indexes[
                                                                             :len_generator_titlels - size_val]

            else:
                raise AttributeError('The "subsampling" parameter should be set to "random" or "crossover".')
