

import matplotlib.pyplot as plt
#save history
import json

list_name = [
                "20_01_2022_new_unet_train_origin_smart_lr_1000ep_6_classes",
                "20_01_2022_new_unet_train_origin_smart_lr_1000ep_5_classes",
                "20_01_2022_new_unet_train_origin_smart_lr_1000ep_1_class"
            ]
            
add_name_graph = ["6 classes", "5 classes", "1 class"]
            
for i,name in enumerate(list_name):
    with open(name + '.json', 'r') as file:
        history = json.load(file)

    plt.subplot(1, 3, i+1)
    
    
    plt.ylim([0, 1])
    # Обучение и проверка точности значений
    plt.plot(history["dice_coef_multilabel"][0:200])
    plt.plot(history["val_dice_coef_multilabel"][0:200])
    plt.title("Model Dice " + add_name_graph[i])
    plt.ylabel("Dice")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="lower right")
    
plt.show()


# Обучение и проверка величины потерь

plt.plot(history["loss"])
plt.plot(history["val_loss"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper left")
plt.show()