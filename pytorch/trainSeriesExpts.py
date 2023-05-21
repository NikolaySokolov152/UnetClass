# этот скрипт запускает последовательно обучение одной конфигурации с разным количеством классов
from trainer import *
import json

def build_argparser_multi():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default = "multi_config_test.json")
    #parser.add_argument('-c', '--config', type=str, default = None)
    args = parser.parse_args()
    return args

def trainMultipleModels(config_file,
                         config_path,
                         models = ["unet",
                                   "tiny_unet_v3",
                                   "mobile_unet",
                                   "Lars76_unet"],
                         funs = []):
    # change_save_suffix = config["save_inform"]["save_suffix_model"]
    # config["move_to_date_folder"] = False

    for model in models:
        print(f"\nLearning '{model}'  model \n")

        # config["save_inform"]["save_suffix_model"] = change_save_suffix + "_" + str(n_class) + "_classes"
        config_file["model"]["type_model"] = model

        if len(funs) > 0:
            funs[0](config_file, config_path, funs=funs[1:])
        else:
            print(trainByConfig(config_file, config_path))

def trainMultipleClasses(config_file,
                         config_path,
                         classes = [6, 5, 1],
                         funs = []):
    # change_save_suffix = config["save_inform"]["save_suffix_model"]
    # config["move_to_date_folder"] = False

    for n_class in classes:
        print(f"\nLearning model with {n_class} classes\n")

        # config["save_inform"]["save_suffix_model"] = change_save_suffix + "_" + str(n_class) + "_classes"
        config_file["train"]["num_class"] = n_class

        new_path = os.path.basename(config_path)[:-5] + "_" + str(n_class) + "_classes.json"

        if len(funs) > 0:
            funs[0](config_file, new_path, funs=funs[1:])
        else:
            print(trainByConfig(config_file, config_path))

def trainMultipleLoss(config_file,
                      config_path,
                      losses = ["BCELoss",
                                "MSELoss",
                                "DiceLoss"],
                      funs = []):
    # change_save_suffix = config["save_inform"]["save_suffix_model"]
    # config["move_to_date_folder"] = False

    for loss in losses:
        print(f"\nLearning model with loss: {loss}\n")

        # config["save_inform"]["save_suffix_model"] = change_save_suffix + "_" + str(n_class) + "_classes"
        config_file["train"]["loss"] = loss

        new_path = os.path.basename(config_path)[:-5] + "_" + str(loss) + ".json"

        if len(funs) > 0:
            funs[0](config_file, new_path, funs=funs[1:])
        else:
            #print("fake work:", new_path)
            print(trainByConfig(config_file, config_path))

def trainMultipleMulticlassLoss(config_file,
                      config_path,
                      losses = ['BCELossMulticlass',
                                'DiceLossMulticlass',
                                'MSELossMulticlass'],
                      funs = []):
    # change_save_suffix = config["save_inform"]["save_suffix_model"]
    # config["move_to_date_folder"] = False

    for loss in losses:
        print(f"\nLearning model with loss: {loss}\n")

        # config["save_inform"]["save_suffix_model"] = change_save_suffix + "_" + str(n_class) + "_classes"
        config_file["train"]["loss"] = loss

        new_path = os.path.basename(config_path)[:-5] + "_" + str(loss) + ".json"

        if len(funs) > 0:
            funs[0](config_file, new_path, funs=funs[1:])
        else:
            #print("fake work:", new_path)
            print(trainByConfig(config_file, config_path))

def trainMultipleActivation(config_file,
                            config_path,
                            last_activations = ["arctan_activation",
                                               "softsign_activation",
                                               "sigmoid_activation",
                                               "linear_activation",
                                               "inv_square_root_activation",
                                               "cdf_activation",
                                               "hardtanh_activation"],
                            funs = []):

    # change_save_suffix = config["save_inform"]["save_suffix_model"]
    # config["move_to_date_folder"] = False

    for last_activation in last_activations:
        print(f"\nLearning model with last_activation: {last_activation}\n")

        # config["save_inform"]["save_suffix_model"] = change_save_suffix + "_" + str(n_class) + "_classes"
        config_file["model"]["last_activation"] = last_activation

        new_path = os.path.basename(config_path)[:-5] + "_" + str(last_activation) + ".json"

        if len(funs) > 0:
            funs[0](config_file, new_path, funs=funs[1:])
        else:
            print(trainByConfig(config_file, config_path))

def trainMultipleOptimazer(config_file,
                           config_path,
                           optimizers = ['Adam',
                                          'AdamW',
                                          'RMSprop',
                                          'NovoGrad'],
                           funs = []):

    # change_save_suffix = config["save_inform"]["save_suffix_model"]
    # config["move_to_date_folder"] = False

    for optimizer in optimizers:
        print(f"\nLearning model with optimizer: {optimizer}\n")

        # config["save_inform"]["save_suffix_model"] = change_save_suffix + "_" + str(n_class) + "_classes"
        config_file["train"]["optimizer"] = optimizer

        new_path = os.path.basename(config_path)[:-5] + "_" + str(optimizer) + "_optimizer.json"

        if len(funs) > 0:
            funs[0](config_file, new_path, funs=funs[1:])
        else:
            print(trainByConfig(config_file, config_path))

def trainMultiple(config_file, config_path, funs = []):
    if len(funs) > 0:
        funs[0](config_file, config_path, funs=funs[1:])
    else:
        print(trainByConfig(config_file, config_path))

def strListTrainMultiple2fun(strList):
    result = []

    for str_train_multiple in strList:
        result.append(globals()[str_train_multiple])

    return result

if __name__ == '__main__':
    args = build_argparser_multi()

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

        list_multi_fun = strListTrainMultiple2fun(multi_config_file["list_experiments"])

        trainMultiple(config_file, config_path, list_multi_fun)
    else:
        print("ERROR OPEN CONFIG")
        raise ValueError("ERROR OPEN CONFIG")
