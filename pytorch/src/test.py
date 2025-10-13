import sys
if not __name__ == "__main__":
    sys.path.append("src/")

import cv2
import json
import numpy as np
import os
import time

#from tqdm import tqdm
from tqdm.auto import tqdm

from pipeliner import Pipeliner
from prepare_data import saveResultMask, tiledGen, prepare_list_batch_to_list_imgs, read_img, to_0_1_format_img
from tilingImages import glit_image, split_image

def type_experiment_parcer(dict_config):
    return dict_config["model"]["experiment_type"]

def model_parcer(dict_config):
    if not dict_config["model"]["type_model"]:
        return "tiny_unet_v3"
    else:
        return dict_config["model"]["type_model"]

def activation_parcer(dict_config):
    # GET LAST ACTIVATION
    if not "last_activation" in dict_config["model"].keys():
        last_activation = 'sigmoid_activation'
    else:
        last_activation = dict_config["model"]["last_activation"]
    return last_activation

def silence_mode_parcer(dict_config):
    # GET LAST ACTIVATION
    if "silence_mode" in dict_config.keys():
        silence_mode = dict_config["silence_mode"]
    else:
        silence_mode = False
    return silence_mode

def device_parcer(dict_config):
    # GET WORKING DEVICE
    if not dict_config["device"]:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        if dict_config["device"].lower() == 'cuda':
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                raise Exception("ERROR! Cuda device no working !")

        elif dict_config["device"].lower() == 'cpu':
            device = 'cpu'
        else:
            print("WARNING! I don't know what is using device, I will use the device as I see fit")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print("Using device:", device)
    return device

def num_class_channel_parcer(dict_config):
    # GET MODEL
    num_channel = 1 if dict_config["img_transform_data"]["color_mode_img"] == 'gray' else 3
    n_classes = num_channel if dict_config["img_transform_data"]["mode_mask"] == "image" else dict_config["train"]["num_class"]
    # Для диффузионки кол-во каналов для входа и выхода одинаковое
    if type_experiment_parcer(dict_config)=="diffusion":
        if dict_config["img_transform_data"]["mode_mask"] == "no_mask":
            n_classes=0
        num_channel=num_channel+n_classes
        n_classes=num_channel

    return n_classes, num_channel

def classnames_parcer(dict_config):
    # для чтения старых конфигов
    if "mask_name_label_list" in dict_config.keys():
        classnames = dict_config["mask_name_label_list"]
    else:
        classnames = dict_config["train"]["mask_name_label_list"]

    return [name.replace(" ", "_") for name in classnames]

def getPipliner(path_dir_to_model, config_name, device=None):
    name_model = "model_by_" + config_name

    pipeliner_path = os.path.join(path_dir_to_model, name_model+"_pipeline.pkl")
    all_model_path = os.path.join(path_dir_to_model, name_model + ".pt")
    model_weights_path = os.path.join(path_dir_to_model, name_model + ".pth")
    if os.path.isfile(pipeliner_path):
        print("Found pipeline file.")
        model = Pipeliner.load_pipeliner(pipeliner_path)
    elif os.path.isfile(all_model_path) or os.path.isfile(model_weights_path):
        with open(os.path.join(path_dir_to_model, config_name + ".json")) as config_buffer:
            config_file = json.load(config_buffer)

        model_class = model_parcer(config_file)
        last_activation = activation_parcer(config_file)
        num_classes, num_channel = num_class_channel_parcer(config_file)

        silence_mode = silence_mode_parcer(config_file)
        type_task = type_experiment_parcer(config_file)

        hidden_params = {}
        use_device = device_parcer(config_file) if device is None else device
        classnames = classnames_parcer(config_file)

        model = Pipeliner(model_class,
                          last_activation,
                          num_classes,
                          num_channel,
                          use_device,
                          silence_mode,
                          type_task,
                          hidden_params,
                          classnames=classnames)

        if os.path.isfile(model_weights_path):
            print("Found model weights file and config.")
            model.load_model_weights_path(model_weights_path)
        else:
            print("Found model file and config.")
            model.load_model_by_path(all_model_path)

    else:
        msg = f"FILE ERROR!!! Model weights data '{name_model}___' don't founded in '{path_dir_to_model}'!!!"
        raise FileNotFoundError(msg)

    return model, name_model

def readPredictDataset(path, as_gray=False):
    list_test_dir = os.listdir(os.path.join(path))
    list_test_img_dir = [name for name in list_test_dir if name.endswith((".png", ".jpg"))]

    dataset = []
    for name in list_test_img_dir:
        img = read_img(os.path.join(path, name), as_gray)
        img = to_0_1_format_img(img)
        dataset.append(([img], [name]))
    return dataset

def glit_mask(tiled_masks, out_size, tile_info, overlap = 64):
    masks = glit_image(tiled_masks, out_size, tile_info, overlap)
    return np.array(masks)

def test_data(model_pipeliner,
              dataset,
              save_mask_dir=None,
              batch_size = 2,
              tiled_data={"size":256, "overlap":64, "unique_area":0},
              save_spliting_dir=None):

    if len(dataset) == 0:
        raise Exception(f"No image to predict")

    if tiled_data is not None:
        size = tiled_data["size"]
        overlap = tiled_data["overlap"]
        unique_area = tiled_data["unique_area"]
        test_mode = "tiled"
    else:
        test_mode = "full"

    print(f'Test mode "{test_mode}"')

    ret_images = []
    ret_names = []

    slices_tqdm = tqdm(dataset, ncols=80, desc="Test", position=0, disable=model_pipeliner.silence_mode)
    for imgs_batch, img_names_batch in slices_tqdm:
        if test_mode == "tiled":
            img_shapes = []
            tile_info_list = []
            list_of_tilled_imgs = []
            ret_images_batch = []

            # распил
            for i, img in enumerate(imgs_batch):
                img_shapes.append(img.shape[:2])
                tiled_arr, tile_info = split_image(img, save_spliting_dir, size, overlap, unique_area)
                tile_info_list.append(tile_info)
                list_of_tilled_imgs += tiled_arr
                #for iterat, img_tile in enumerate(tiled_arr):
                #    cv2.imshow(f"img {iterat}", img_tile)
                #cv2.waitKey()
            img_generator = tiledGen(list_of_tilled_imgs, batch_size=batch_size)
            results = prepare_list_batch_to_list_imgs(model_pipeliner.predict(img_generator))

            # сборка
            for i in range(len(imgs_batch)):
                one_img_counts = tile_info_list[i]
                masks = results[:one_img_counts[0]*one_img_counts[1]] # взять картинки на один слой
                ret_images_batch.append(glit_mask(masks, img_shapes[i], one_img_counts, overlap))
                results = results[one_img_counts[0]*one_img_counts[1]:] # изъять взятые картинки

        elif test_mode == "full":
            img_generator = tiledGen(imgs_batch, batch_size=batch_size)
            ret_images_batch = prepare_list_batch_to_list_imgs(model_pipeliner.predict(img_generator))
        else:
            msg = f"ERROR! Preparation mode {test_mode} is not implemented."
            raise Exception(msg)
        # print("glit_mask", res_img.shape)

        if save_mask_dir is not None:
            saveResultMask(save_mask_dir, ret_images_batch, img_names_batch, classnames=model_pipeliner.classnames)
        ret_images += ret_images_batch
        ret_names += img_names_batch

    return ret_images, ret_names
