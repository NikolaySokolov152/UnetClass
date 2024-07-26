import os.path

###############################################################################################################################
from src.train import *
#from src.diffusion_train import *
from src.dataGenerator import *
from src.models import *
from src.metrics import *
from src.lr_scheduler import *

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

########################################################## доделать чтение дифузионных конфигов

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
    parser.add_argument('-c', '--config', type=str, default="img2img/config_test.json")
    #parser.add_argument('-c', '--config', type=str, default = "segmentation/config_test.json")
    #parser.add_argument('-c', '--config', type=str, default = None)
    parser.add_argument('-s', '--silence_mode', action='store_true')

    parser.add_argument('-uc', '--user_constrol', action='store_true',
                        help="Enabling change settings mode using command line arguments")

    parser.add_argument('-lr_s', '--lr_scheduler', type=str, default=None,
                        help="Selecting the learning rate change function (default config setting or standart)",
                        choices=['standart',
                                 'lr_scheduler_loss_mix',
                                 'lr_scheduler_loss'])

    parser.add_argument('-la', '--last_activation', type=str, default=None,
                        help="Selecting the last activation change function (default config setting or sigmoid_activation)",
                        choices=["arctan_activation",
                                 "softsign_activation",
                                 "sigmoid_activation",
                                 "linear_activation",
                                 "inv_square_root_activation",
                                 "cdf_activation",
                                 "hardtanh_activation"])

    parser.add_argument('--seed', type=int, default=None,
                        help='Selecting the initialization of the random number generator (default config setting or 42).')

    parser.add_argument('-m', '--model', type=str, default=None,
                        help='Selecting the model (default config setting or tiny_unet_v3).',
                        choices=["unet",
                                 "tiny_unet",
                                 "tiny_unet_v3",
                                 "mobile_unet",
                                 "Lars76_unet"])

    parser.add_argument('-optm', '--optimizer', type=str, default=None,
                        help='Selecting the optimizer (default config setting or Adam).',
                        choices=['Adam',
                                 'AdamW',
                                 'RMSprop',
                                 'NovoGrad'])

    parser.add_argument('-loss', '--losses', default=None, nargs='+',
                        help='Selecting the list losses (default config setting or DiceLossMulticlass).',
                        choices = ['DiceLoss',
                                   'BCELoss',
                                   'MSELoss',
                                   'DiceLossMulticlass',
                                   'BCELossMulticlass',
                                   'MSELossMulticlass',
                                   'LossDistance2Nearest'])

    parser.add_argument('-classes', '--num_classes', type=int, default=None,
                        help="Selecting the number classes (default config setting or 6)")

    parser.add_argument('-n', '--num_epochs', type=int, default=None,
                        help='Selecting the number epochs (default config setting).')

    parser.add_argument('-b', '--batch_size', type=int, default=None,
                        help="Selecting the batch_size (default config setting)")

    parser.add_argument('-name', '--add_save_name', type=str, default=None,
                        help="Added prefix save model name")

    args = parser.parse_args()
    return args


def type_experiment_parcer(dict_config):
    return dict_config["model"]["experiment_type"]

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

def set_cofig_seed(dict_config):
    # SET SEED
    if "train" in dict_config.keys() and "seed" in dict_config["train"].keys():
        seed_all(dict_config["train"]["seed"])
    else:
        seed_all(42)
        dict_config["train"]["seed"] = 42

def generator_parcer(dict_config, device, silence_mode=False):
    # GET DATA FOR GENERATOR
    augmentation = dict_config["augmentation"]

    if type(dict_config["data_info"]) is dict:
        dir_data = InfoDirData(**dict_config["data_info"])
    elif type(dict_config["data_info"]) is list:
        dir_data = []
        for dataset_info in dict_config["data_info"]:
            dir_data.append(InfoDirData(**dataset_info))
    else:
        raise Exception(f"ERROR don't know data type 'data_info':  {type(dict_config['data_info'])}")

    transform_data = TransformData(**dict_config["img_transform_data"])
    # для чтения старых конфигов
    if not "batch_size" in dict_config["img_transform_data"].keys():
        transform_data.batch_size = dict_config["train"]["batch_size"]
    # для перестраховки
    if dict_config["img_transform_data"]["mode_mask"] == "image":
        transform_data.binary_mask=False

    save_inform = SaveData(**dict_config["save_inform"])

    if not "type_load_data" in dict_config["generator_config"].keys():
        dict_config["generator_config"]['type_load_data'] = 'img'

    # для чтения старых конфигов
    if "mask_name_label_list" in dict_config.keys():
        classnames = dict_config["mask_name_label_list"]
    else:
        classnames = dict_config["train"]["mask_name_label_list"]

    # GET DATA GENERATOR
    if not dict_config["generator_config"]["type_gen"] or\
       dict_config["generator_config"]["type_gen"] == "default" or\
       dict_config["generator_config"]["type_gen"] == "all_reader":
            myGen = DataGeneratorReaderAll(dir_data = dir_data,
                                           num_classes     = dict_config["train"]["num_class"],
                                           mode            = dict_config["generator_config"]["mode"],
                                           aug_dict        = augmentation,
                                           list_class_name = classnames,
                                           augment         = dict_config["generator_config"]["augment"],
                                           tailing         = dict_config["generator_config"]["tailing"],
                                           shuffle         = dict_config["generator_config"]["shuffle"],
                                           seed            = dict_config["generator_config"]["seed"],
                                           subsampling     = dict_config["generator_config"]["subsampling"],
                                           transform_data  = transform_data,
                                           save_inform     = save_inform,
                                           share_validat   = dict_config["generator_config"]["share_validat"],
                                           type_load_data  = dict_config["generator_config"]['type_load_data'],
                                           silence_mode    = silence_mode,
                                           device          = device)
    else:
        print("GEN CHOISE ERROR: now you can only choose: 'default' ('all_reader') generator")
        raise AttributeError("GEN CHOISE ERROR: now you can only choose: 'default' ('all_reader') generator")
    return myGen

def divece_model_optimizer_parcer(dict_config):
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

    # GET MODEL
    num_channel = 1 if dict_config["img_transform_data"]["color_mode_img"] == 'gray' else 3
    n_classes = num_channel if dict_config["img_transform_data"]["mode_mask"] == "image" else dict_config["train"]["num_class"]
    # Для диффузионки кол-во каналов для входа и выхода одинаковое
    if type_experiment_parcer(dict_config)=="diffusion":
        if dict_config["img_transform_data"]["mode_mask"] == "no_mask":
            n_classes=0
        num_channel=num_channel+n_classes
        n_classes=num_channel

    if "pretrained_weights" in dict_config["train"] and dict_config["train"]["pretrained_weights"] is not None:
        model = torch.load(dict_config["train"]["pretrained_weights"])
    else:
        using_model = ["unet", "tiny_unet", "tiny_unet_v3", "mobile_unet", "Lars76_unet"]

        if not dict_config["model"]["type_model"] or dict_config["model"]["type_model"] == "tiny_unet_v3":
            if not dict_config["model"]["type_model"]:
                dict_config["model"]["type_model"] = "tiny_unet_v3"
            model = Tiny_unet_v3(n_classes=n_classes,
                                 n_channels=num_channel)
        elif dict_config["model"]["type_model"] == "unet":
            model = UNet(n_classes=n_classes,
                         n_channels=num_channel)
        elif dict_config["model"]["type_model"] == "tiny_unet":
            model = Tiny_unet(n_classes=n_classes,
                              n_channels=num_channel)
        elif dict_config["model"]["type_model"] == "mobile_unet":
            model = MobileUNet(n_classes=n_classes,
                               n_channels=num_channel)
        elif dict_config["model"]["type_model"] == "Lars76_unet":
            # из статьи "Effect of the output activation function on the probabilities and errors in medical image segmentation"
            # https://arxiv.org/pdf/2109.00903.pdf
            model = Lars76_unet(n_classes=n_classes,
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
    return device, model, optimizer

def losses_parcer(dict_config):
    # GET LOSSES
    using_loss = ['DiceLoss', 'BCELoss', 'MSELoss', 'DiceLossMulticlass', 'BCELossMulticlass', 'MSELossMulticlass', 'LossDistance2Nearest', "HuberLoss"]

    losses = []
    if not "loss" in dict_config["train"].keys():
        losses.append("DiceLossMulticlass")
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
    return losses

def metrics_parcer(dict_config):
    # GET METRICS ###################################################################################################################
    if type_experiment_parcer(dict_config)=="diffusion":
        num_channel = 1 if dict_config["img_transform_data"]["color_mode_img"] == 'gray' else 3
        n_classes = num_channel if dict_config["img_transform_data"]["mode_mask"] == "image" else\
                              0 if dict_config["img_transform_data"]["mode_mask"] == "no_mask" else\
                    dict_config["train"]["num_class"]
        num_class=num_channel+n_classes
    else:
        num_class = dict_config["train"]["num_class"]

    metrics=[
            #Dice(),
            DiceMultilabel(num_class)
            ]
    return metrics

def num_epochs_parcer(dict_config):
    # GET NUM EPOCHS
    num_epochs = dict_config["train"]["num_epochs"]
    return num_epochs

def model_name_parcer(dict_config):
    # GET SAVE MODEL NAME
    modelName = ""

    if len(dict_config["save_inform"]["save_prefix_model"]) > 0:
        modelName += dict_config["save_inform"]["save_prefix_model"] + "_"
        # str(dict_config["img_transform_data"]["target_size"][0]) + "_" +\

    modelName += dict_config["model"]["type_model"]  # + "_"  +\
    # str(dict_config["train"]["num_class"]) + "_num_class" #+ "_" +\
    # dict_config["model"]["optimizer"]

    if len(dict_config["save_inform"]["save_suffix_model"]) > 0:
        modelName += "_" + dict_config["save_inform"]["save_suffix_model"]

    return modelName


 ################################################################################################################### Доделать
def diffusion_config_parcer(dict_config):
    diffusion_config = dict_config["model"]["diffusion_config"]
    return diffusion_config

def config_parcer(dict_config):
    device, model, optimizer = divece_model_optimizer_parcer(dict_config)
    set_cofig_seed(dict_config)
    last_activation = activation_parcer(dict_config)
    silence_mode = silence_mode_parcer(dict_config)
    myGen = generator_parcer(dict_config, device, silence_mode)
    num_epochs = num_epochs_parcer(dict_config)
    lr_scheduler = lr_scheduler_parcer(dict_config)
    losses = losses_parcer(dict_config)
    metrics = metrics_parcer(dict_config)

    type_task_train = type_experiment_parcer(dict_config)
    train_args = None
    if type_task_train == "diffusion":
        train_args=diffusion_config_parcer(dict_config)

    # DEBUGGING TRAIN LOADER
    if dict_config["debug_mode"]:
        model_name = model_name_parcer(dict_config)
        print("Type task:", type_task_train)

        print("print Train randoms seed:", dict_config["train"]["seed"])
        print ("print myGen:")
        print ("\ttypeGen:", myGen.typeGen)
        print ("\tdir_data:", myGen.dir_data)
        print ("\tlist_class_name:", myGen.list_class_name)
        print ("\tnum_classes:", myGen.num_classes)
        print ("\ttransform_data:",
               ", color_mode_img", myGen.transform_data.color_mode_img,
               ", mode_mask", myGen.transform_data.mode_mask,
               ", target_size", myGen.transform_data.target_size,
               ", batch_size",  myGen.transform_data.batch_size,
               ", mask_binary_mode", myGen.transform_data.binary_mask,
               ", normalization_img_fun", myGen.transform_data.normalization_img_fun,
               ", normalization_mask_fun", myGen.transform_data.normalization_mask_fun)
        print ("\taug_dict:", myGen.aug_dict)
        print ("\tmode:", myGen.mode)
        print ("\tsubsampling:", myGen.subsampling)
        print ("\tsave_inform:",
               " save_to_dir ", myGen.save_inform.save_to_dir,
               " save_prefix_image ", myGen.save_inform.save_prefix_image,
               " save_prefix_mask ", myGen.save_inform.save_prefix_mask)
        print ("\tshare_validat:", myGen.share_val)
        print ("\taugment:", myGen.augment)
        print ("\tshuffle:", myGen.shuffle)
        print ("\tseed:", myGen.seed)
        print ("\ttailing:", myGen.tailing)
        print ("\tlen list_img_name:", len(myGen.list_img_name))
        print("\tlr_scheduler:", lr_scheduler.__name__)
        print()

        print ("print Model:")
        print ("\tnum_epochs:", num_epochs)
        print("\tmodel:", model.__class__.__name__)
        print ("\tmodelName:", model_name)
        print ("\toptimizer:", optimizer)
        print ("\tModelSize:", sum(p.numel() for p in model.parameters()))
        print("losses:", losses)
        print("using device:", device)
        print("last_activation:", last_activation)
        print()

        print("Silence_mode", silence_mode)

    return myGen,\
           model,\
           last_activation,\
           num_epochs,\
           device,\
           optimizer,\
           metrics,\
           losses,\
           lr_scheduler,\
           silence_mode,\
           type_task_train,\
           train_args

def trainByConfig(config_file, path_config, retrain = False):
    data_save = None
    if (config_file["move_to_date_folder"]):
        if "experiment_data" in config_file.keys() and config_file["experiment_data"] is not None:
            data_save = config_file["experiment_data"]
        else:
            now = datetime.datetime.now()
            data_save = f"{now.year:04}_{now.month:02}_{now.day:02}"

    model_name_parc = model_name_parcer(config_file)

    path_config = os.path.basename(path_config)[:-5] + "_" + model_name_parc + ".json"
    setproctitle.setproctitle(os.path.basename(path_config)[:-5])
    modelName = "model_by_" + os.path.basename(path_config)[:-5]

    type_task_train = type_experiment_parcer(config_file)

    if not os.path.isdir(type_task_train):
        print(f"create dir:'{type_task_train}'")
        os.mkdir(type_task_train)
    type_with_data_save = os.path.join(type_task_train, data_save)

    if (not retrain) and os.path.isdir(type_with_data_save) and os.path.isfile(os.path.join(type_with_data_save, modelName + '.pt')):
        return f"{modelName} already trained"

    print(f"start train model '{modelName}' and save in '{type_with_data_save}'")

    # при запуске нескольких экспериментов забивается память
    with torch.no_grad():
        torch.cuda.empty_cache()
    gc.collect()

    myGen,\
    model,\
    last_activation,\
    num_epochs,\
    device,\
    optimizer,\
    losses,\
    metrics,\
    lr_scheduler,\
    silence_mode,\
    type_task_train,\
    train_args=config_parcer(config_file)

    #history = fit_diffusion_Model(myGen,
    history = fitModel(myGen,
                       model,
                       last_activation,
                       num_epochs,
                       device,
                       optimizer,
                       losses,
                       metrics,
                       modelName,
                       lr_scheduler=lr_scheduler,
                       silence_mode=silence_mode,
                       train_mode=type_task_train,
                       train_args=train_args)

    try:
        history['lr'] = np.array(history['lr']).astype(float).tolist()
    except:
        print("WARNING: no lr")

    with open("history_" + modelName + '.json', 'w') as file:
        json.dump(history, file, indent=4)

    if config_file is not None and (config_file["move_to_date_folder"]):
        if not os.path.isdir(type_with_data_save):
            print(f"create dir:'{type_with_data_save}'")
            os.mkdir(type_with_data_save)
        print(f"move model, history and config in '{type_with_data_save}'")

        with open(os.path.join(type_with_data_save, os.path.basename(path_config)), 'w') as file:
            json.dump(config_file, file, indent=4)
        shutil.move(modelName + '.pt', os.path.join(type_with_data_save, modelName + '.pt'))
        shutil.move("history_" + modelName + '.json', os.path.join(type_with_data_save, "history_" + modelName + '.json'))
    return f"End experiment: {modelName}"

if __name__ == '__main__':
    print("parse")
    args = build_argparser()
    silence_mode = args.silence_mode

    if args.config:
        path = args.config
        with open(args.config) as config_buffer:
            print("open config")
            config = json.loads(config_buffer.read())

        if not silence_mode:
            if "silence_mode" in config.keys():
                silence_mode = config["silence_mode"]
        config["silence_mode"] = silence_mode

        if args.user_constrol:
            print("\tuse console argument")
            if args.lr_scheduler is not None:
                print(f"\t\tuse console lr_scheduler: {args.lr_scheduler}")
                config["train"]["lr_scheduler"] = args.lr_scheduler

            if args.seed is not None:
                print(f"\t\tuse console seed: {args.seed}")
                config["train"]["seed"] = args.seed
                config["generator_config"]["seed"] = args.seed
                path = os.path.basename(path)[:-5] + f"_seed_{args.seed}.json"

            if args.optimizer is not None:
                print(f"\t\tuse console optimizer: {args.optimizer}")
                config["train"]["optimizer"] = args.optimizer

            if args.losses is not None:
                print(f"\t\tuse console losses: {args.losses}")
                config["train"]["loss"] = args.losses

                if type(args.losses) is list:
                    str_loss = "_".join(args.losses)
                else:
                    str_loss = args.losses
                path = os.path.basename(path)[:-5] + "_" + str(str_loss) + ".json"

            if args.num_epochs is not None:
                print(f"\t\tuse console num_epochs: {args.num_epochs}")
                config["train"]["num_epochs"] = args.num_epochs

            if args.batch_size is not None:
                print(f"\t\tuse console batch_size: {args.batch_size}")
                config["train"]["batch_size"] = args.batch_size

            if args.num_classes is not None:
                print(f"\t\tuse console num_classes: {args.num_classes}")
                config["train"]["num_class"] = args.num_classes

            if args.add_save_name is not None:
                print(f"\t\tuse console add_save_name: {args.add_save_name}")
                config["save_inform"]["save_prefix_model"] += args.add_save_name

            if args.last_activation is not None:
                print(f"\t\tuse console last_activation: {args.last_activation}")
                config["model"]["last_activation"] = args.last_activation

                path = os.path.basename(path)[:-5] + "_" + str(args.last_activation) + ".json"

            if args.model is not None:
                print(f"\t\tuse console model: {args.model}")
                config["model"]["type_model"] = args.model

            print()
    else:
        config = None
        path = None
        print("ERROR CONFIG")

    print(trainByConfig(config, path))

    #viewerLearningRate.viewData(history.history)
