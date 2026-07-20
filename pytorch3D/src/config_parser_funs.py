import sys
if not __name__ == "__main__":
    sys.path.append("src/")
import torch

from losses import AVAILABLE_LOSSES
from dataGenerator3D import DataGeneratorReaderAll, InfoDirData, CommonTransformData, SaveGeneratorData

def type_experiment_parcer(dict_config):
    if "experiment_type" in dict_config["model"].keys():
        return dict_config["model"]["experiment_type"]
    else:
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

def generator_parcer(dict_config, silence_mode=False):
    device=device_parcer(dict_config)

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

    transform_data = CommonTransformData(**dict_config["img_transform_data"])
    # для чтения старых конфигов
    if not "batch_size" in dict_config["img_transform_data"].keys():
        transform_data.batch_size = dict_config["train"]["batch_size"]
    # для перестраховки
    if dict_config["img_transform_data"]["type_load_target"] in ["image", "nii"]:
        transform_data.binary_mask=False

    save_inform = SaveGeneratorData(**dict_config["save_inform"])

    classnames = classnames_parcer(dict_config)

    num_class, _ = num_class_channel_parcer(dict_config)

    # GET DATA GENERATOR
    if not dict_config["generator_config"]["type_gen"] or\
       dict_config["generator_config"]["type_gen"] == "default" or\
       dict_config["generator_config"]["type_gen"] == "all_reader":
            myGen = DataGeneratorReaderAll(dir_data = dir_data,
                                           num_classes     = num_class,
                                           mode            = dict_config["generator_config"]["mode"],
                                           aug_dict        = augmentation,
                                           list_class_name = classnames,
                                           is_augment= dict_config["generator_config"]["augment"],
                                           is_shuffle= dict_config["generator_config"]["shuffle"],
                                           #seed            = dict_config["generator_config"]["seed"],
                                           subsampling     = dict_config["generator_config"]["subsampling"],
                                           transform_data  = transform_data,
                                           save_inform     = save_inform,
                                           share_validat   = dict_config["generator_config"]["share_validat"],
                                           is_silence_mode= silence_mode,
                                           is_calculate_statistic= dict_config["balancing_parameters"]["calculate_statistic"] if "balancing_parameters" in dict_config.keys() else False,
                                           device          = device)
    else:
        print("GEN CHOISE ERROR: now you can only choose: 'default' ('all_reader') generator")
        raise AttributeError("GEN CHOISE ERROR: now you can only choose: 'default' ('all_reader') generator")
    return myGen

def num_class_channel_parcer(dict_config):
    # GET MODEL
    if "num_class" in dict_config["model"].keys():
        n_classes = dict_config["model"]["num_class"]
        num_channel = dict_config["model"]["num_channel"]
    else:
        num_channel = 1 if dict_config["img_transform_data"]["color_mode_img"] == 'gray' else 3
        n_classes = num_channel if dict_config["img_transform_data"]["mode_mask"] == "image" else dict_config["train"]["num_class"]
        # Для диффузионки кол-во каналов для входа и выхода одинаковое
        if type_experiment_parcer(dict_config)=="diffusion":
            if dict_config["img_transform_data"]["mode_mask"] == "no_mask":
                n_classes=0
            num_channel=num_channel+n_classes
            n_classes=num_channel

    return n_classes, num_channel

def model_parcer(dict_config):
    if not dict_config["model"]["type_model"]:
        return "tiny_unet_v3"
    else:
        return dict_config["model"]["type_model"]

def losses_parcer(dict_config):
    # GET LOSSES

    losses = []
    if not "loss" in dict_config["train"].keys():
        losses.append("DiceLossMulticlass")
    else:
        try:
            if type(dict_config["train"]["loss"]) is not list:
                if dict_config["train"]["loss"] in AVAILABLE_LOSSES:
                    losses = [dict_config["train"]["loss"]]
                else:
                    raise AttributeError(f"unknown loss: '{dict_config['train']['loss']}'!")
            else:
                for loss in dict_config["train"]["loss"]:
                    if not loss in AVAILABLE_LOSSES:
                        raise AttributeError(f"unknown loss: '{loss}'!")

                losses = dict_config["train"]["loss"]

            if len(losses) == 0:
                raise AttributeError("losses is clear !")
        except Exception as ex:
            str_using_loss = "' ,'".join(AVAILABLE_LOSSES)
            print(f"LOSS CHOICE ERROR: now you can only choose: '{str_using_loss}' loss. " + str(ex))
            raise AttributeError(f"LOSS CHOICE ERROR: now you can only choose: '{str_using_loss}' loss. " + str(ex))
    return losses

def metrics_parcer(dict_config):
    metrics = []
    if "metrics" in dict_config["train"].keys():
        metrics = dict_config["train"]["metrics"]
    else:
        metrics = ["Dice",
                   #"DiceMultilabel",
                  "MultiMetricClasses"]
    return metrics

def num_epochs_parcer(dict_config):
    # GET NUM EPOCHS
    return dict_config["train"]["num_epochs"]

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

def optimizer_parcer(dict_config):
    # GET OPTIMIZER
    using_optimizer = ['Adam', 'AdamW', 'RMSprop', 'NovoGrad']

    if not "optimizer" in dict_config["train"].keys():
        return "Adam"
    else:
        return dict_config["train"]["optimizer"]

def description_parcer(dict_config):
    return dict_config["experiment_description"] if "experiment_description" in dict_config.keys() else None

 ################################################################################################################### Доделать
def diffusion_config_parcer(dict_config):
    diffusion_config = dict_config["model"]["diffusion_config"]
    return diffusion_config

def statistics_params_parser(dict_config):
    if "balancing_parameters" in dict_config.keys():
        return {"balancing_parameters": dict_config["balancing_parameters"]}
    else:
        return {}

def classnames_parcer(dict_config):
    if "mask_name_label_list" in dict_config["model"].keys():
        classnames = dict_config["model"]["mask_name_label_list"]

    # для чтения старых конфигов
    elif "mask_name_label_list" in dict_config.keys():
        classnames = dict_config["mask_name_label_list"]
    else:
        classnames = dict_config["train"]["mask_name_label_list"]

    return [name.replace(" ", "_") for name in classnames]
