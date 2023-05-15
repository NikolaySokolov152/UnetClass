from src.train import *
from src.dataGenerator import *
from src.models import *
from src.metrics import *

import torch
import torch_optimizer as optim_mod
import torch.optim as optim

import setproctitle

#import viewerLearningRate

import argparse
import datetime
import json
import numpy as np
import shutil
import gc

########################################################## добавить метрики по классам

def seed_all(seed):
    torch.manual_seed(seed)
    # might not be needed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default = "config_test.json")
    #parser.add_argument('-c', '--config', type=str, default = None)
    args = parser.parse_args()
    return args

# функция изменения lr
def lr_scheduler(epoch):
    if epoch < 100:
        return 0.0001
    elif epoch < 125:
        return 0.00005
    elif epoch < 150:
        return 0.00001
    elif epoch < 175:
        return 0.000005
    return 0.000001

def config_parcer(dict_config):
    # GET WORKING DEVICE
    if not dict_config["device"]:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        if dict_config["device"].lower() == 'cuda':
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                print("WARNING! Cuda device no working, I using CPU!")
        elif dict_config["device"].lower() == 'cpu':
            device = 'cpu'
        else:
            print("WARNING! I don't know what is using device, I will use the device as I see fit")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print("Using device:", device)

    if not "last_activation" in dict_config["model"].keys():
        last_activation = 'sigmoid_activation'
    else:
        last_activation = dict_config["model"]["last_activation"]


    # GET DATA FOR GENERATOR
    augmentation = dict_config["augmentation"]

    if not augmentation["rotation_range"]:
        augmentation["rotation_range"] = 0
    if not augmentation["width_shift_range"]:
        augmentation["width_shift_range"] = 0
    if not augmentation["height_shift_range"]:
        augmentation["height_shift_range"] = 0
    if not augmentation["zoom_range"]:
        augmentation["zoom_range"] = 0
    if not augmentation["horizontal_flip"]:
        augmentation["horizontal_flip"] = False
    if not augmentation["vertical_flip"]:
        augmentation["vertical_flip"] = False
    if not augmentation["noise_limit"]:
        augmentation["noise_limit"] = 0
    if not augmentation["fill_mode"]:
        augmentation["fill_mode"] = 0

    dir_data = InfoDirData(dir_img_name = dict_config["train"]["dir_img_path"],
               dir_mask_name            = dict_config["train"]["dir_mask_path_without_name"],
               add_mask_prefix          = dict_config["data_info"]["add_mask_prefix"])

    transform_data = TransformData(color_mode_img = dict_config["img_transform_data"]["color_mode_img"],
                                   mode_mask      = dict_config["img_transform_data"]["mode_mask"],
                                   target_size    = dict_config["img_transform_data"]["target_size"],
                                   batch_size     = dict_config["train"]["batch_size"])

    save_inform = SaveData(save_to_dir       = dict_config["save_inform"]["save_to_dir"],
                           save_prefix_image = dict_config["save_inform"]["save_prefix_image"],
                           save_prefix_mask  = dict_config["save_inform"]["save_prefix_mask"])

    # SET SEED
    if "train" in dict_config.keys() and "seed" in dict_config["train"].keys():
        seed_all(dict_config["train"]["seed"])
    else:
        seed_all(42)
        dict_config["train"]["seed"] = 42

    # GET DATA GENERATOR
    if not dict_config["generator_config"]["type_gen"] or dict_config["generator_config"]["type_gen"] == "default":
        myGen = DataGenerator(dir_data = dir_data,
                          num_classes     = dict_config["train"]["num_class"],
                          mode            = dict_config["generator_config"]["mode"],
                          aug_dict        = augmentation,
                          list_class_name = dict_config["train"]["mask_name_label_list"],
                          augment         = dict_config["generator_config"]["augment"],
                          tailing         = dict_config["generator_config"]["tailing"],
                          shuffle         = dict_config["generator_config"]["shuffle"],
                          seed            = dict_config["generator_config"]["seed"],
                          subsampling     = dict_config["generator_config"]["subsampling"],
                          transform_data  = transform_data,
                          save_inform     = save_inform,
                          share_validat   = dict_config["generator_config"]["share_validat"])
    elif dict_config["generator_config"]["type_gen"] == "all_reader":
            myGen = DataGeneratorReaderAll(dir_data = dir_data,
                          num_classes     = dict_config["train"]["num_class"],
                          mode            = dict_config["generator_config"]["mode"],
                          aug_dict        = augmentation,
                          list_class_name = dict_config["train"]["mask_name_label_list"],
                          augment         = dict_config["generator_config"]["augment"],
                          tailing         = dict_config["generator_config"]["tailing"],
                          shuffle         = dict_config["generator_config"]["shuffle"],
                          seed            = dict_config["generator_config"]["seed"],
                          subsampling     = dict_config["generator_config"]["subsampling"],
                          transform_data  = transform_data,
                          save_inform     = save_inform,
                          share_validat   = dict_config["generator_config"]["share_validat"])
    else:
        print("GEN CHOISE ERROR: now you can only choose: 'default', 'all_reader' generator")
        raise AttributeError("GEN CHOISE ERROR: now you can only choose: 'default', 'all_reader' generator")

    # GET MODEL
    num_channel = 1 if dict_config["img_transform_data"]["color_mode_img"] == 'gray' else 3

    if "pretrained_weights" in dict_config["train"] and dict_config["train"]["pretrained_weights"] is not None:
        model = torch.load(dict_config["train"]["pretrained_weights"])
    else:
        using_model = ["unet", "tiny_unet", "tiny_unet_v3", "mobile_unet", "Lars76_unet"]

        if not dict_config["model"]["type_model"] or dict_config["model"]["type_model"] == "unet":
            if not dict_config["model"]["type_model"]:
                dict_config["model"]["type_model"] = "unet"
            model = UNet(n_classes = dict_config["train"]["num_class"],
                         n_channels= num_channel)
        elif dict_config["model"]["type_model"] == "tiny_unet":
            model = Tiny_unet(n_classes = dict_config["train"]["num_class"],
                              n_channels= num_channel)
        elif dict_config["model"]["type_model"] == "tiny_unet_v3":
            model = Tiny_unet_v3(n_classes = dict_config["train"]["num_class"],
                                 n_channels= num_channel)
        elif dict_config["model"]["type_model"] == "mobile_unet":
            model = MobileUNet(n_classes = dict_config["train"]["num_class"],
                               n_channels= num_channel)
        elif dict_config["model"]["type_model"] == "Lars76_unet":
            # из статьи "Effect of the output activation function on the probabilities and errors in medical image segmentation"
            # https://arxiv.org/pdf/2109.00903.pdf
            model = Lars76_unet(n_classes  =dict_config["train"]["num_class"],
                                n_channels=num_channel)
        else:
            str_using_model = "' ,'".join(using_model)
            print(f"MODEL CHOICE ERROR: now you can only choose: '{str_using_model}' model")
            raise AttributeError(f"MODEL CHOICE ERROR: now you can only choose: '{str_using_model}' model")

    model.to(device)

    # GET OPTIMIZER
    using_optimizer = ['Adam', 'AdamW', 'RMSprop', 'NovoGrad']

    if not "optimizer" in dict_config["train"].keys() or dict_config["train"]["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
    elif dict_config["train"]["optimizer"] == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    elif dict_config["train"]["optimizer"] == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=5e-3)
    elif dict_config["train"]["optimizer"] == "NovoGrad":
        optimizer = optim_mod.NovoGrad(model.parameters(), lr=1e-3, weight_decay=1e-3)
    else:
        str_using_optimizer = "' ,'".join(using_optimizer)
        print(f"OPTIMIZER CHOICE ERROR: now you can only choose: '{str_using_optimizer}' optimizer")
        raise AttributeError(
            f"OPTIMIZER CHOICE ERROR: now you can only choose: '{str_using_optimizer}' optimizer")

    # GET LOSSES
    using_loss = ['DiceLoss', 'BCELoss', 'MSELoss', 'DiceLossMulticlass', 'BCELossMulticlass', 'MSELossMulticlass']

    losses = []
    if not "loss" in dict_config["train"].keys():
        losses.append("DiceLoss")
    else:
        try:
            if type(dict_config["train"]["loss"]) is not list:
                if dict_config["train"]["loss"] in using_loss:
                    losses = [dict_config["train"]["loss"]]
                else:
                    raise AttributeError(f"unknown loss: '{dict_config['train']['loss']}'!")
            else:
                for loss in dict_config["train"]["loss"]:
                    if not loss in using_loss:
                        raise AttributeError(f"unknown loss: '{loss}'!")

                losses = dict_config["train"]["loss"]

            if len(losses) == 0:
                raise AttributeError("losses is clear !")
        except Exception as ex:
            str_using_loss = "' ,'".join(using_loss)
            print(f"LOSS CHOICE ERROR: now you can only choose: '{str_using_loss}' loss. " + str(ex))
            raise AttributeError(f"LOSS CHOICE ERROR: now you can only choose: '{str_using_loss}' loss. " + str(ex))

    # GET METRICS
    #metrics = [universal_dice_coef_multilabel(dict_config["train"]["num_class"]),
    #           universal_dice_coef_multilabel_arr(dict_config["train"]["num_class"])]
    metrics = [Dice(), DiceMultilabel(dict_config["train"]["num_class"])]

    # GET NUM EPOCHS
    num_epochs = dict_config["train"]["num_epochs"]

    # GET SAVE MODEL NAME
    modelName = ""

    if len(dict_config["save_inform"]["save_prefix_model"]) > 0:
        modelName += dict_config["save_inform"]["save_prefix_model"] + "_"
                     #str(dict_config["img_transform_data"]["target_size"][0]) + "_" +\

    modelName += dict_config["model"]["type_model"] #+ "_"  +\
                 # str(dict_config["train"]["num_class"]) + "_num_class" #+ "_" +\
                 #dict_config["model"]["optimizer"]

    if len(dict_config["save_inform"]["save_suffix_model"]) > 0:
        modelName += "_" + dict_config["save_inform"]["save_suffix_model"]

    # DEBUGGING TRAIN LOADER
    if dict_config["debug_mode"]:
        print("print Train randoms seed:",dict_config["train"]["seed"])
        print ("print myGen:")
        print ("\ttypeGen:", myGen.typeGen)
        print ("\tdir_data:", "dir_img_name", myGen.dir_data.dir_img_name, ", dir_mask_name", myGen.dir_data.dir_mask_name, ", add_mask_prefix",  myGen.dir_data.add_mask_prefix)
        print ("\tlist_class_name:", myGen.list_class_name)
        print ("\tnum_classes:", myGen.num_classes)
        print ("\ttransform_data:", ", color_mode_img",  myGen.transform_data.color_mode_img, ", mode_mask", myGen.transform_data.mode_mask, ", target_size", myGen.transform_data.target_size, ", batch_size",  myGen.transform_data.batch_size)
        print ("\taug_dict:", myGen.aug_dict)
        print ("\tmode:", myGen.mode)
        print ("\tsubsampling:", myGen.subsampling)
        print ("\tsave_inform:", " save_to_dir ", myGen.save_inform.save_to_dir, " save_prefix_image ", myGen.save_inform.save_prefix_image, " save_prefix_mask ", myGen.save_inform.save_prefix_mask)
        print ("\tshare_validat:", myGen.share_val)
        print ("\taugment:", myGen.augment)
        print ("\tshuffle:", myGen.shuffle)
        print ("\tseed:", myGen.seed)
        print ("\ttailing:", myGen.tailing)
        print ("\tlen list_img_name:", len(myGen.list_img_name))
        print()

        print ("print Model:")
        print ("\tnum_epochs:", num_epochs)
        print ("\tmodelName:", modelName)
        print ("\toptimizer:", optimizer)
        print ("\tModelSize:", sum(p.numel() for p in model.parameters()))
        print("losses:", losses)
        print("using device:", device)
        print("last_activation:", last_activation)
        print()

    return myGen, model, last_activation, num_epochs, device, optimizer, metrics, losses, modelName

def trainByConfig(config_file, path_config):
    # при запуске нескольких экспериментов забивается память
    with torch.no_grad():
        torch.cuda.empty_cache()
    gc.collect()

    myGen, model, last_activation, num_epochs, device, optimizer, losses, metrics, modelName = config_parcer(config_file)

    path_config = os.path.basename(path_config)[:-5] + "_" + modelName + ".json"
    setproctitle.setproctitle(os.path.basename(path_config)[:-5])

    modelName = "model_by_" + os.path.basename(path_config)[:-5]

    data_save = None
    if (config_file["move_to_date_folder"]):
        if "experiment_data" in config_file.keys() and config_file["experiment_data"] is not None:
            data_save = config_file["experiment_data"]
        else:
            now = datetime.datetime.now()
            data_save = f"{now.year:04}_{now.month:02}_{now.day:02}"

    history = fitModel(myGen, model, last_activation, num_epochs, device, optimizer, losses, metrics, modelName, lr_scheduler = lr_scheduler)

    try:
        history['lr'] = np.array(history['lr']).astype(float).tolist()
    except:
        print("WARNING: no lr")

    with open("history_" + modelName + '.json', 'w') as file:
        json.dump(history, file, indent=4)

    if config_file is not None and (config_file["move_to_date_folder"]):
        if not os.path.isdir(data_save):
            print(f"create dir:'{data_save}'")
            os.mkdir(data_save)
        print(f"move model, history and config in '{data_save}'")

        with open(os.path.join(data_save, os.path.basename(path_config)), 'w') as file:
            json.dump(config_file, file, indent=4)
        shutil.move(modelName + '.pt', os.path.join(data_save, modelName + '.pt'))
        shutil.move("history_" + modelName + '.json', os.path.join(data_save, "history_" + modelName + '.json'))

if __name__ == '__main__':
    args = build_argparser()
    if args.config:
        path = args.config
        with open(args.config) as config_buffer:
            config = json.loads(config_buffer.read())
    else:
        config = None
        path = None

    trainByConfig(config, path)

    #viewerLearningRate.viewData(history.history)
