# этот скрипт запускает последовательно обучение одной конфигурации с разным количеством классов
from trainer import *
import numpy as np
import json
import sys

def build_argparser_multi():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default = "segmentation/multi_config_test.json")
    #parser.add_argument('-c', '--config', type=str, default = None)
    parser.add_argument('-s', '--silence_mode', action='store_true')
    parser.add_argument('-seed', '--random_seed', type=int, default=None)

    args = parser.parse_args()
    return args

def printSeriesFunction():
    print("'trainMultipleModels' - choice models\n",
          "'trainDataset' - choice dataset by indexes list\n",
          "'trainSeries' - change seed\n",
          "'trainMultipleDataProportion' - change dataset proportion\n",
          "'trainMultipleOptimazer' - choice optimazers\n"
          "'trainMultipleActivation' - choice fun last activations\n",
          "'trainMultipleLoss' - choice losses\n",
          "'trainMultipleClasses' - change number classes"
          )

def trainMultipleModels(config_file,
                        config_path,
                        models = ["unet",
                                  #"tiny_unet",
                                  "tiny_unet_v3",
                                  "mobile_unet",
                                  "Lars76_unet"],
                        funs = [],
                        args_exp=None,
                        tabs=""):
    # change_save_suffix = config["save_inform"]["save_suffix_model"]
    # config["move_to_date_folder"] = False

    for model in models:
        print(f"\n{tabs}Learning '{model}'  model\n")

        # config["save_inform"]["save_suffix_model"] = change_save_suffix + "_" + str(n_class) + "_classes"
        config_file["model"]["type_model"] = model

        if len(funs) > 0:
            if args_exp is None:
                funs[0](config_file, config_path, funs=funs[1:], args_exp=args_exp, tabs=tabs+"\t")
            else:
                funs[0](config_file, config_path, args_exp[0], funs=funs[1:], args_exp=args_exp[1:], tabs=tabs+"\t")
        else:
            print(trainByConfig(config_file, config_path))

def trainMultipleClasses(config_file,
                         config_path,
                         classes = [6, 5, 1],
                         funs = [],
                         args_exp=None,
                         tabs=""):
    # change_save_suffix = config["save_inform"]["save_suffix_model"]
    # config["move_to_date_folder"] = False

    for n_class in classes:
        print(f"\n{tabs}Learning model with {n_class} classes\n")

        # config["save_inform"]["save_suffix_model"] = change_save_suffix + "_" + str(n_class) + "_classes"
        config_file["train"]["num_class"] = n_class

        new_path = os.path.basename(config_path)[:-5] + f"_{n_class}_classes.json"

        if len(funs) > 0:
            if args_exp is None:
                funs[0](config_file, new_path, funs=funs[1:], args_exp=args_exp, tabs=tabs+"\t")
            else:
                funs[0](config_file, new_path, args_exp[0], funs=funs[1:], args_exp=args_exp[1:], tabs=tabs+"\t")
        else:
            print(trainByConfig(config_file, new_path))

def trainMultipleLoss(config_file,
                      config_path,
                      losses = ["BCELoss",
                                "MSELoss",
                                "DiceLoss"],
                      funs = [],
                      args_exp=None,
                      tabs=""):
    # change_save_suffix = config["save_inform"]["save_suffix_model"]
    # config["move_to_date_folder"] = False

    for loss in losses:
        print(f"\n{tabs}Learning model with loss: {loss}\n")

        # config["save_inform"]["save_suffix_model"] = change_save_suffix + "_" + str(n_class) + "_classes"
        config_file["train"]["loss"] = loss

        if type(loss) is list:
            str_loss = "_".join(loss)
        else:
            str_loss = loss
        new_path = os.path.basename(config_path)[:-5] + f"_{str_loss}.json"

        if len(funs) > 0:
            if args_exp is None:
                funs[0](config_file, new_path, funs=funs[1:], args_exp=args_exp, tabs=tabs+"\t")
            else:
                funs[0](config_file, new_path, args_exp[0], funs=funs[1:], args_exp=args_exp[1:], tabs=tabs+"\t")
        else:
            #print("fake work:", new_path)
            print(trainByConfig(config_file, new_path))

def trainMultipleActivation(config_file,
                            config_path,
                            last_activations = ["arctan_activation",
                                               "softsign_activation",
                                               "sigmoid_activation",
                                               "linear_activation",
                                               "inv_square_root_activation",
                                               "cdf_activation",
                                               "hardtanh_activation"],
                            funs = [],
                            args_exp=None,
                            tabs=""):

    # change_save_suffix = config["save_inform"]["save_suffix_model"]
    # config["move_to_date_folder"] = False

    for last_activation in last_activations:
        print(f"\n{tabs}Learning model with last_activation: {last_activation}\n")

        # config["save_inform"]["save_suffix_model"] = change_save_suffix + "_" + str(n_class) + "_classes"
        config_file["model"]["last_activation"] = last_activation

        new_path = os.path.basename(config_path)[:-5] + f"_{last_activation}.json"

        if len(funs) > 0:
            if args_exp is None:
                funs[0](config_file, new_path, funs=funs[1:], args_exp=args_exp, tabs=tabs+"\t")
            else:
                funs[0](config_file, new_path, args_exp[0], funs=funs[1:], args_exp=args_exp[1:], tabs=tabs+"\t")
        else:
            print(trainByConfig(config_file, new_path))

def trainMultipleOptimazer(config_file,
                           config_path,
                           optimizers = ['Adam',
                                          'AdamW',
                                          'RMSprop',
                                          'NovoGrad'],
                           funs = [],
                           args_exp=None,
                           tabs=""):

    # change_save_suffix = config["save_inform"]["save_suffix_model"]
    # config["move_to_date_folder"] = False

    for optimizer in optimizers:
        print(f"\n{tabs}Learning model with optimizer: {optimizer}\n")

        # config["save_inform"]["save_suffix_model"] = change_save_suffix + "_" + str(n_class) + "_classes"
        config_file["train"]["optimizer"] = optimizer

        new_path = os.path.basename(config_path)[:-5] + f"_{optimizer}_optimizer.json"

        if len(funs) > 0:
            if args_exp is None:
                funs[0](config_file, new_path, funs=funs[1:], args_exp=args_exp, tabs=tabs+"\t")
            else:
                funs[0](config_file, new_path, args_exp[0], funs=funs[1:], args_exp=args_exp[1:], tabs=tabs+"\t")
        else:
            print(trainByConfig(config_file, new_path))

def trainMultipleDataProportion(config_file,
                                config_path,
                                proportions = [val/10 for val in range(1, 11)],
                                funs = [],
                                args_exp=None,
                                tabs=""):

    # change_save_suffix = config["save_inform"]["save_suffix_model"]
    # config["move_to_date_folder"] = False

    if type(proportions[0]) is list:
        proportions = zip(*proportions)

    for proportion in proportions:
        print(f"\n{tabs}Learning model with data proportional: {proportion}\n")

        if type(config_file["data_info"]) is list:
            if type(proportion) is float or type(proportion) is int:
                config_file["data_info"][0]["proportion_of_dataset"] = proportion
            else:
                for i, dataset_info in enumerate(config_file["data_info"]):
                    dataset_info["proportion_of_dataset"] = proportion[i]
        else:
            config_file["data_info"]["proportion_of_dataset"] = proportion
        # config["save_inform"]["save_suffix_model"] = change_save_suffix + "_" + str(n_class) + "_classes"


        new_path = os.path.basename(config_path)[:-5] + f"_{proportion}_data_proportional.json"

        if len(funs) > 0:
            if args_exp is None:
                funs[0](config_file, new_path, funs=funs[1:], args_exp=args_exp, tabs=tabs+"\t")
            else:
                funs[0](config_file, new_path, args_exp[0], funs=funs[1:], args_exp=args_exp[1:], tabs=tabs+"\t")
        else:
            print(trainByConfig(config_file, new_path))

def trainSeries(config_file,
                config_path,
                num_experiments,
                funs = [],
                args_exp=None,
                tabs=""):
    # change_save_suffix = config["save_inform"]["save_suffix_model"]
    # config["move_to_date_folder"] = False

    seed_list = []
    for id_experiment in range(num_experiments):

        random_seed = np.random.randint(np.iinfo(np.int32).max)
        while random_seed in seed_list:
            random_seed = np.random.randint(np.iinfo(np.int32).max)
        seed_list.append(random_seed)

        print(f"\n{tabs}Learning model with seed {random_seed}\n")

        # config["save_inform"]["save_suffix_model"] = change_save_suffix + "_" + str(n_class) + "_classes"
        config_file["train"]["seed"] = random_seed

        new_path = os.path.basename(config_path)[:-5] + f"_seed_{random_seed}.json"

        if len(funs) > 0:
            if args_exp is None:
                funs[0](config_file, new_path, funs=funs[1:], args_exp=args_exp, tabs=tabs+"\t")
            else:
                funs[0](config_file, new_path, args_exp[0], funs=funs[1:], args_exp=args_exp[1:], tabs=tabs+"\t")
        else:
            print(trainByConfig(config_file, new_path))


def trainDataset(config_file,
                 config_path,
                 turtle_list_indexes_and_names=[],
                 funs = [],
                 args_exp=None,
                 tabs=""):
    # change_save_suffix = config["save_inform"]["save_suffix_model"]
    # config["move_to_date_folder"] = False

    datasets_all = config_file["data_info"]

    for name, list_indexes in turtle_list_indexes_and_names:
        print(f"\n{tabs}Learning model with dataset {name}\n")

        config_file["data_info"] = [datasets_all[i] for i in list_indexes]

        new_path = os.path.basename(config_path)[:-5] + f"_dataset_{name}.json"

        if len(funs) > 0:
            if args_exp is None:
                funs[0](config_file, new_path, funs=funs[1:], args_exp=args_exp, tabs=tabs+"\t")
            else:
                funs[0](config_file, new_path, args_exp[0], funs=funs[1:], args_exp=args_exp[1:], tabs=tabs+"\t")
        else:
            print(trainByConfig(config_file, new_path))

    config_file["data_info"] = datasets_all

def trainMultiple(config_file, config_path, funs=[], args_exp=None, tabs=""):
    if len(funs) > 0:
        if args_exp is None:
            funs[0](config_file, config_path, funs=funs[1:], args_exp=args_exp, tabs=tabs)
        else:
            funs[0](config_file, config_path, args_exp[0], funs=funs[1:], args_exp=args_exp[1:], tabs=tabs)
    else:
        print(trainByConfig(config_file, config_path))

def strListTrainMultiple2fun(strList):
    result = []

    for str_train_multiple in strList:
        result.append(globals()[str_train_multiple])

    return result

if __name__ == '__main__':
    args = build_argparser_multi()
    silence_mode = args.silence_mode
    if args.config:
        multi_config_path = args.config
        with open(multi_config_path) as multi_config_buffer:
            multi_config_file = json.load(multi_config_buffer)

        config_path = multi_config_file["config_path"]
        with open(config_path) as config_buffer:
            config_file = json.load(config_buffer)

        if (config_file["move_to_date_folder"]):
            if not "experiment_data" in multi_config_file.keys() or multi_config_file["experiment_data"] is None:
                if not "experiment_data" in config_file.keys() or config_file["experiment_data"] is None:
                    now = datetime.datetime.now()
                    config_file["experiment_data"] = f"{now.year:04}_{now.month:02}_{now.day:02}"
                    print("Use now data", config_file["experiment_data"])
            else:
                config_file["experiment_data"] = multi_config_file["experiment_data"]

        if not silence_mode:
            if "silence_mode" in multi_config_file.keys():
                silence_mode = multi_config_file["silence_mode"]
            elif "silence_mode" in config_file.keys():
                silence_mode = config_file["silence_mode"]

        config_file["silence_mode"] = silence_mode

        if args.random_seed is not None:
            config_file["train"]["seed"] = args.random_seed
            config_path = os.path.basename(config_path)[:-5] + f"_seed_{args.random_seed}.json"

        list_multi_fun = strListTrainMultiple2fun(multi_config_file["list_experiments"])

        if "list_arguments" in multi_config_file.keys():
            args_exp = multi_config_file["list_arguments"]
        else:
            args_exp = None

        trainMultiple(config_file, config_path, funs=list_multi_fun, args_exp=args_exp)
    else:
        print("ERROR OPEN CONFIG")
        raise ValueError("ERROR OPEN CONFIG")
