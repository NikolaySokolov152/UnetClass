import numpy as np
import os
import tqdm
from src.dataGenerator import InfoDirData
import random
import time
import skimage.io as io
import sys
import cv2

mode_mask = 'separated'
save_path = "data/np_data_train"
color_mode_img = "gray"
num_classes = 6
binary_mask = True

list_class_name = ["mitochondria", "PSD", "vesicles", "axon", "boundaries", "mitochondrial_boundaries"]

dir_data = [InfoDirData(dir_img_name="C:/Users/Sokol-PC/Synthetics/dataset/test_dataset1/original",
                           dir_mask_name="C:/Users/Sokol-PC/Synthetics/dataset/test_dataset1/",
                           add_mask_prefix='',
                           proportion_of_dataset=0.1),

            InfoDirData(dir_img_name = "G:/Data/Unet_multiclass/data/cutting data/original",
                           dir_mask_name ="G:/Data/Unet_multiclass/data/cutting data/",
                           add_mask_prefix = '',
                           proportion_of_dataset=0.01)]

img_type = ('.png', '.jpg', '.jpeg')

def to_0_1_format_img(in_img):
    max_val = in_img[:, :].max()
    if max_val <= 1:
        return in_img
    else:
        out_img = in_img / 255
        return out_img

def load_new_data():
    imgs = []
    masks_glob = []
    list_img_name = []

    if type(dir_data) is InfoDirData:
        list_InfoDirData = [dir_data]
    elif type(dir_data) is list or type(dir_data) is tuple:
        list_InfoDirData = dir_data
    else:
        raise Exception(f"ERROR type dir_data don't know: {dir_data}")

    for i, info_dir_data in enumerate(list_InfoDirData):
        dataset_imgs = []
        dataset_masks_glob = []

        print(f"Load {i + 1} dataset of {len(list_InfoDirData)}")

        if not os.path.exists(info_dir_data.dir_img_name):
            print("Image path not found!")
            raise AttributeError(f"Image path '{info_dir_data.dir_img_name}' not found!")

        img_names_all = [name for name in os.listdir(info_dir_data.dir_img_name) if name.endswith((img_type))]
        #len_names = len(img_names_all)
        #work_num_img = int(round(len_names * info_dir_data.proportion_of_dataset))

        #print(
        #    f"I take {work_num_img} of {len_names} imagess, proportion of dataset: {info_dir_data.proportion_of_dataset}")
        img_names = img_names_all #random.sample(img_names_all, work_num_img)

        list_img_name += img_names

        time.sleep(0.2)  # чтобы tqdm не печатал вперед print
        for name in tqdm(img_names, file=sys.stdout, desc='\tLoad slices'):
            img_path = os.path.join(info_dir_data.dir_img_name, name)

            if color_mode_img == "rgb":
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

        if mode_mask == 'separated':
            for index_name, name in enumerate(
                    tqdm(img_names, file=sys.stdout, desc='\tLoad masks', disable=self.silence_mode)):

                class_mask = np.zeros((*dataset_imgs[index_name].shape[:2], num_classes), np.float32)

                for i in range(num_classes):
                    img_path = os.path.join(info_dir_data.dir_mask_name, list_class_name[i],
                                            info_dir_data.add_mask_prefix + name)

                    if os.path.isfile(img_path):
                        image = io.imread(img_path, as_gray=True)
                        masks = image.astype(np.float32)
                        masks = to_0_1_format_img(masks)

                        if binary_mask:
                            masks[masks[:, :] < 0.5] = 0
                            masks[masks[:, :] > 0.4] = 1
                    else:
                        print("no open ", img_path)
                        masks = np.zeros(dataset_imgs[index_name].shape[:2], np.float32)

                    class_mask[:, :, i] = masks
                dataset_masks_glob.append(class_mask)

        elif mode_mask == 'image':
            for index_name, name in enumerate(
                    tqdm(img_names, file=sys.stdout, desc='\tLoad res image')):
                img_path = os.path.join(info_dir_data.dir_mask_name, list_class_name[0],
                                        info_dir_data.add_mask_prefix + name)
                if os.path.isfile(img_path):
                    if color_mode_img == "rgb":
                        img = io.imread(img_path)
                    elif color_mode_img == "hsv":
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

        elif mode_mask == 'no_mask':
            # заглушка
            for index_name, name in enumerate(
                    tqdm(img_names, file=sys.stdout, desc='Create fake masks')):
                dataset_masks_glob.append(np.zeros(dataset_imgs[index_name].shape[:2], np.float32))
        else:
            raise AttributeError(
                'The "mode_mask" parameter should be set to "separated", "image" or "no_mask" otherwise not implemented.')

        imgs += dataset_imgs
        masks_glob += dataset_masks_glob

    return list_img_name, imgs, masks_glob

def saveNpyData(path, list_img_name, all_imges, all_masks):
    if not os.path.isdir(path):
        print(f"create dir:'{path}'")
        os.makedirs(path)

    save_img_path = os.path.join(path, "original")
    save_img_arr = np.array((list_img_name, all_imges), dtype=object)
    np.save(save_img_path + '.npy', save_img_arr)

    if mode_mask == 'separated' or \
       mode_mask == 'image':
        save_mask_path = os.path.join(path, "mask")
        save_mask_arr = np.array((list_img_name, all_masks), dtype=object)
        np.save(save_mask_path + '.npy', save_mask_arr)


list_img_name, all_imges, all_masks = load_new_data()
saveNpyData(save_path, list_img_name, all_imges, all_masks)
print(f"save {len(list_img_name)} images")