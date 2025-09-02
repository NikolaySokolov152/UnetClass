import argparse
import datetime
import gc
import json
import os.path
import numpy as np
import random
import setproctitle
import shutil
import torch
import logging

#import viewerLearningRate

from src.config_parser_funs import (type_experiment_parcer,
                                    activation_parcer,
                                    silence_mode_parcer,
                                    generator_parcer,
                                    num_class_channel_parcer,
                                    model_parcer,
                                    losses_parcer,
                                    metrics_parcer,
                                    num_epochs_parcer,
                                    model_name_parcer,
                                    optimizer_parcer,
                                    diffusion_config_parcer,
                                    statistics_params_parser,
                                    description_parcer)
from src.lr_scheduler import *
from src.pipeliner import Pipeliner

########################################################## добавить метрики по классам
########################################################## доделать чтение дифузионных конфигов

def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="segmentation/config_test.json")
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

def set_cofig_seed(dict_config):
    # SET SEED
    if "train" in dict_config.keys() and "seed" in dict_config["train"].keys():
        seed_all(dict_config["train"]["seed"])
    else:
        seed_all(42)
        dict_config["train"]["seed"] = 42

def config_parcer(dict_config):
    set_cofig_seed(dict_config)

    model_class=model_parcer(dict_config)
    last_activation=activation_parcer(dict_config)
    num_classes, num_channel=num_class_channel_parcer(dict_config)

    silence_mode=silence_mode_parcer(dict_config)
    type_task=type_experiment_parcer(dict_config)

    myGen = generator_parcer(dict_config, silence_mode)
    num_epochs = num_epochs_parcer(dict_config)
    lr_scheduler = lr_scheduler_parcer(dict_config)
    losses = losses_parcer(dict_config)
    metrics_names = metrics_parcer(dict_config)
    optimizer_name = optimizer_parcer(dict_config)

    model_name = model_name_parcer(dict_config)
    device = myGen.device

    train_args=statistics_params_parser(dict_config)

    description=description_parcer(dict_config)

    hidden_params = {}
    if type_task == "diffusion":
        hidden_params["diffusion_args"]=diffusion_config_parcer(dict_config)

    # DEBUGGING TRAIN LOADER
    if dict_config["debug_mode"]:
        print("Type task:", type_task)
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
        print ("\taugment:", myGen.is_augment)
        print ("\tshuffle:", myGen.is_shuffle)
        #print ("\tseed:", myGen.seed)
        print ("\tlen list_img_name:", len(myGen.list_img_name))
        print("\tlen generator:", len(myGen))
        print("\tlr_scheduler:", lr_scheduler.__name__)

        print()
        print("print Model:")
        print("\tnum_epochs:", num_epochs)
        print("\ttype_model:", model_class)
        print("\tmodelName:", model_name)
        print("\tnum_classes:", num_classes)
        print("\tnum_channel:", num_channel)
        print("\tlast_activation:", last_activation)
        print("\thidden_params:", hidden_params)
        print("\tmetrics_names:", metrics_names)
        print()
        print ("train Params:")
        print("\tlosses:", losses)
        print("\tusing device:", device)
        print("\tadd_train_args:", train_args)
        print("\toptimizer_name:", optimizer_name)
        print("Silence_mode", silence_mode)
        print()

    return (model_class,
            last_activation,
            num_classes,
            num_channel,
            device,
            silence_mode,
            type_task,
            hidden_params,

            myGen,
            num_epochs,
            optimizer_name,
            metrics_names,
            losses,
            lr_scheduler,
            train_args,
            description)


def trainByConfig(config_file, path_config, retrain = False):
    try:
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
        #print(f"type_task_train: {type_task_train}")
        if (not retrain) and os.path.isdir(type_with_data_save) and os.path.isfile(os.path.join(type_with_data_save, modelName + '.pt')):
            return f"{modelName} already trained"
        print(f"start train model '{modelName}' and save in '{type_with_data_save}'")

        # при запуске нескольких экспериментов забивается память
        with torch.no_grad():
            torch.cuda.empty_cache()
        gc.collect()

        (model_class,
        last_activation,
        num_classes,
        num_channel,
        device,
        silence_mode,
        type_task,
        hidden_params,

        myGen,
        num_epochs,
        optimizer_name,
        metric_names,
        losses,
        lr_scheduler,
        train_args,
        description)=config_parcer(config_file)

        #################################################################################################################################
        use_validation = True
        use_train_metric = True
        save_model_mode = "weights_only_after_finish"  # [ "all", "weights_only", "weights_only_after_finish"]
        save_pipeliner_mode = "save_pipeliner"  # ["no_save", "save_pipeliner"]


        model = Pipeliner(model_class,
                          last_activation,
                          num_classes,
                          num_channel,
                          device,
                          silence_mode,
                          type_task,
                          hidden_params,
                          description)

        history = model.train(myGen,
                              num_epochs,
                              optimizer_name,
                              metric_names,
                              losses,
                              lr_scheduler,
                              use_validation,
                              use_train_metric,
                              train_args,
                              model_name=modelName,
                              device=device,
                              save_model_mode=save_model_mode,
                              save_pipeliner_mode=save_pipeliner_mode)


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

            if os.path.isfile(modelName + "_pipeline.pkl"):
                shutil.move(modelName + "_pipeline.pkl", os.path.join(type_with_data_save, modelName + "_pipeline.pkl"))

            if os.path.isfile(modelName + '.pt'):
                shutil.move(modelName + '.pt', os.path.join(type_with_data_save, modelName + '.pt'))
            elif os.path.isfile(modelName + '.pth'):
                shutil.move(modelName + '.pth', os.path.join(type_with_data_save, modelName + '.pth'))

            shutil.move("history_" + modelName + '.json', os.path.join(type_with_data_save, "history_" + modelName + '.json'))

        logging.basicConfig(filename="experiments_log.log",filemode="a+")
        logging.info(f'Successful completion of the experiment "{modelName}" and save in "{type_with_data_save}"')
        return f"End experiment: {modelName}"

    except Exception as ex:
        logging.basicConfig(filename="experiments_log.log", filemode="a+")
        logging.error(f'Experiment "{modelName}" ended with an error: "{ex.__str__()}"')
        raise ex

def changer_arg_config_and_path(args, config, path, main_dict, arg_value, dict_val_name=None):
    if dict_val_name is None:
        dict_val_name = arg_value
    set_val = getattr(args, arg_value, None)
    if set_val is None:
        return path
    else:
        print(f"\t\tuse console {arg_value}: {set_val}")
        if isinstance(main_dict, list):
            for dict_name in main_dict:
                config[dict_name][dict_val_name] = set_val
        else:
            config[main_dict][dict_val_name] = set_val

        if isinstance(set_val, list):
            str_vals = "_".join(set_val)
        else:
            if arg_value == "model" or arg_value == "add_save_name":
                return path
            else:
                str_vals = set_val

        return f"{os.path.basename(path)[:-5]}_{str_vals}.json"

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

            ### Раздел конфига (где менять)     Имя в args из cmd    Имя в конфиге (None значит как в args)
            list_check_to_change_val = [
                ("train",                       "lr_scheduler",      None),
                (["train", "generator_config"], "seed",              None),
                ("train",                       "optimizer",         None),
                ("train",                       "loss",              None),
                ("train",                       "num_epochs",        None),
                ("train",                       "batch_size",        None),
                ("train",                       "num_class",         None),
                ("save_inform",                 "save_prefix_model", None),
                ("model",                       "last_activation",   None),
                ("model",                       "model",             "type_model"),
                ("save_inform",                 "add_save_name",     "save_suffix_model")
            ]
            for main_dict, value, dict_val_name in list_check_to_change_val:
                path = changer_arg_config_and_path(args, config, path, main_dict, value, dict_val_name)

            print()
    else:
        config = None
        path = None
        print("ERROR CONFIG")

    print(trainByConfig(config, path))

    #viewerLearningRate.viewData(history.history)
