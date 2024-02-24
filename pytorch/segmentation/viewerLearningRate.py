import json
import matplotlib.pyplot as plt
import os

import argparse

def viewData(history, name = "", save_path = None, view = True, print_save_model_point = True):

    fig = plt.figure(figsize=(20, 12), dpi=100)
    fig.suptitle("model: " + name, fontsize=16)
    
    # рисование точки сохранения модели
    if print_save_model_point:
        if not 'model_saving_epoch' in history.keys():
            print_save_model_point = False
        else:
            save_epoch = history['model_saving_epoch']
    count_epoch = len(history["train_work_time"])

    # metric 1
    ax = fig.add_subplot(2, 2, 1)
    ax.set_ylim(ymax = 1.0)
    ax.plot(range(1, count_epoch+1), history["metrics"]['Dice'])
    ax.plot(range(1, count_epoch+1), history["val_metrics"]["Dice"])
    if print_save_model_point:
        ax.plot(save_epoch, history["val_metrics"]["Dice"][save_epoch-1], '-ro', label='save model point')
    ax.set_title("Model Dice", fontsize=12)
    ax.set_ylabel("Dice")
    ax.set_xlabel("Epoch")
    ax.legend(["Train", "Validation"], loc="upper left")

    # metric 2
    ax = fig.add_subplot(2, 2, 2)
    ax.set_ylim(ymax = 1.0)
    ax.plot(range(1, count_epoch+1), history["metrics"]["DiceMultilabel"])
    ax.plot(range(1, count_epoch+1), history["val_metrics"]["DiceMultilabel"])
    if print_save_model_point:
        ax.plot(save_epoch, history["val_metrics"]["DiceMultilabel"][save_epoch-1], '-ro', label='save model point')
    ax.set_title("Model DiceMultilabel", fontsize=12)
    ax.set_ylabel("DiceMultilabel")
    ax.set_xlabel("Epoch")
    ax.legend(["Train", "Validation"], loc="upper left")
    
    # loss
    ax = fig.add_subplot(2, 2, 3)
    ax.set_ylim(ymin = 0, ymax = max(history["loss"] + history["val_loss"]))
    
    ax.plot(range(1, count_epoch+1), history["loss"])
    ax.plot(range(1, count_epoch+1), history["val_loss"])
    if print_save_model_point:
        ax.plot(save_epoch,  history["val_loss"][save_epoch-1], '-ro', label='save model point')
    ax.set_title("Loss", fontsize=12)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    ax.legend(["Train", "Validation"], loc="upper left")

    # lr
    try:
        ax = fig.add_subplot(2, 2, 4)
        ax.plot(range(1, count_epoch+1), history["lr"])
        if print_save_model_point:
            ax.plot(save_epoch, history["lr"][save_epoch-1], '-ro', label='save model point')
        ax.set_title("Learning rate", fontsize=12)
        ax.set_ylabel("Lr")
        ax.set_xlabel("Epoch")
    except:
        pass
    
    fig.tight_layout()

    if save_path is not None:
        plt.savefig(os.path.join(save_path, name + ".png"))
    if view:
        plt.show()


def readViewData(name, path, save_path, view = True):
    with open(os.path.join(path, name + '.json'), 'r') as file:
        history = json.load(file)
    #plt.ylim([0, 1])
    # Обучение и проверка точности значений
    viewData(history, name[8:], save_path, view)

def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--path', type=str, default="./")
    args = parser.parse_args()
    return args

def getLossActivationDirs():
    str_data1 = "Lars_test"
    str_data2 = "2023_05_20"
    types_datasets = ["real", "mix", "sint"]

    return_dir = []
    for dataset in types_datasets:
        return_dir.append("_".join([str_data1, dataset, str_data2]))
    return return_dir

def getStandartTestDirs():
    str_data1 = "Models_and_classes_multiclass_BCG"
    str_data2 = "2023_05_21"

    types_datasets = ["real",
                      "mix",
                      "sint",
                      #"sint_v2"
                      ]

    return_dir = []
    for dataset in types_datasets:
        return_dir.append("_".join([str_data1, dataset, str_data2]))
    return return_dir


if __name__ == "__main__":
    #args = build_argparser()
    #paths = [args.path]
    #paths = ['Models_and_classes_sint_2023_05_10']

    #paths = getStandartTestDirs() + getLossActivationDirs() + ["Models_and_classes_sint_v2_2023_05_15"]
    #paths = ["new_loss_dist_test"] #+ getStandartTestDirs()
    paths = ["Test_v10_dataset",
             "Test_mix_v2_dataset"]

    for i, path in enumerate(paths):
        print(f"In the work {i+1} of {len(paths)} dirs named '{path}'")

        save_path = os.path.join(path, "ViewLearning")

        if not os.path.isdir(save_path):
            print(f"create dir:'{save_path}'")
            os.mkdir(save_path)

        list_name = os.listdir(path)
        json_history_name = [name[:-5] for name in list_name if name.endswith(".json") and name.startswith("history_")]

        for history in json_history_name:
            try:
                readViewData(history, path, save_path, view=False)
            except Exception as e:
                print(e)
                pass
