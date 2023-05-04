from src.test import *
from src.npMetrics import *
import os

################################################ сделать с конфиг файлом
################################################ доделать запись метрик в файл
################################################ доделать test.py

def test_models_all_dir():
    str_data = "2023_05_02"

    classnames = ["mitochondria", "PSD", "vesicles", "axon", "boundaries", "mitochondrial boundaries"]

    CNN_name = [
        "model_by_config_sint_Lars76_unet",
        "model_by_config_Lars76_unet"
    ]

    list_CNN_name = []
    for name in CNN_name:
        #change_name = "обучение " + str_data + "/" + name + ".pt"
        change_name = str_data + "/" + name + ".pt"
        list_CNN_name.append(change_name)

    list_CNN_num_class = [
        6,
        6
    ]

    result_CNN_dir = []

    for i in range(len(CNN_name)):
        save_name = "data/result/" + str_data + "/" + str(list_CNN_num_class[i]) + "_class/" + CNN_name[i]
        result_CNN_dir.append(save_name)

    overlap_list = [128]

    # путь до тестовых картинок
    filepath = "G:/Data/Unet_multiclass/data/test"

    list_test_img_dir = os.listdir(os.path.join(filepath))
    list_test_img_dir = [name for name in list_test_img_dir if name.endswith(".png")]

    for i in range(len(list_CNN_num_class)):
        print("predict ", list_CNN_name[i], " model")
        for overlap in overlap_list:
            print("     predict tiled with overlap: ", overlap)

            dataset = {'filenames': list_test_img_dir, "filepath": filepath, "classnames": classnames}
            tiled_data = {"size": 256, "overlap": overlap, "unique_area": 0}

            test_tiled(model_path=list_CNN_name[i],
                       num_class=list_CNN_num_class[i],
                       save_mask_dir=result_CNN_dir[i] + "_" + str(overlap),
                       dataset = dataset,
                       tiled_data=tiled_data,
                       )  # , save_dir= "data/split test/")

    CalulateMetricsDirs(CNN_name,
                        list_CNN_num_class,
                        str_data_time=str_data,
                        overlap=overlap,
                        using_metrics = [Jaccard, Dice],
                        merge_images = True
                        )
    CalulateMetricsDirs(CNN_name,
                        list_CNN_num_class,
                        str_data_time=str_data,
                        overlap=overlap,
                        using_metrics = [Jaccard, Dice],
                        merge_images=False
                        )

if __name__ == "__main__":
    test_models_all_dir()
