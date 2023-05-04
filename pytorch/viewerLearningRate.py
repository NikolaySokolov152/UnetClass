import json
import matplotlib.pyplot as plt
import os

def viewData(history, name = "", view = True):

    fig = plt.figure(figsize=(20, 12), dpi=100)
    
    fig.suptitle("model: " + name, fontsize=16)
    ax = fig.add_subplot(2, 2, 1)
    
    ax.set_ylim(ymax = 1.0)
    
    ax.plot(history["dice_coef"])
    ax.plot(history["val_dice_coef"])
    ax.set_title("Model Dice", fontsize=12)
    ax.set_ylabel("Dice")
    ax.set_xlabel("Epoch")
    ax.legend(["Train", "Validation"], loc="upper left")
    

    ax = fig.add_subplot(2, 2, 2)
    
    ax.set_ylim(ymin = 0)
    ax.plot(history["loss"])
    ax.plot(history["val_loss"])
    ax.set_title("Loss", fontsize=12)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    ax.legend(["Train", "Validation"], loc="upper left")
    
    try:
        ax = fig.add_subplot(2, 2, 3)
        ax.plot(history["lr"])
        ax.set_title("Learning rate", fontsize=12)
        ax.set_ylabel("Lr")
        ax.set_xlabel("Epoch")
    except:
        pass
    
    fig.tight_layout()
    
    if view:
        plt.show()
    else:
        plt.savefig(name + ".png")


def readViewData(name, view = True):
    with open(name + '.json', 'r') as file:
        history = json.load(file)

    #plt.ylim([0, 1])
    # Обучение и проверка точности значений
    viewData(history, name[8:], view)
    

if __name__ == "__main__":
    list_name = os.listdir("./")
    json_history_name = [name[:-5] for name in list_name if name.endswith(".json")]
    
    for history in json_history_name:
        try:
            readViewData(history, False)
        except Exception as e:
            print(e)
            pass
