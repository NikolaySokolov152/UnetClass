from src.test import *
from src.npMetrics import *
import os

################################################ сделать с конфиг файлом
################################################ доделать test.py
################################################ сделать тестирование без сохранения картинок
################################################ разобраться с двойным прогресс баром

def test_models_all_dir(str_data,
                        classnames,
                        list_CNN_num_class,
                        CNN_name,
                        overlap_list,
                        file_test_path,
                        last_activations=None,
                        save_report_path="data/report/",
                        using_metrics=[Jaccard, Dice],
                        only_excel_file=True):

    list_CNN_name = []
    for name in CNN_name:
        #change_name = "обучение " + str_data + "/" + name + ".pt"
        change_name = str_data + "/" + name + ".pt"
        list_CNN_name.append(change_name)

    result_CNN_dir = []

    for i in range(len(CNN_name)):
        save_name = "data/result/" + str_data + "/" + str(list_CNN_num_class[i]) + "_class/" + CNN_name[i]
        result_CNN_dir.append(save_name)

    list_test_img_dir = os.listdir(os.path.join(file_test_path))
    list_test_img_dir = [name for name in list_test_img_dir if name.endswith((".png", ".jpg"))]

    for i in range(len(list_CNN_num_class)):
        print(f"predict model '{list_CNN_name[i]}'")
        for overlap in overlap_list:
            print("     predict tiled with overlap: ", overlap)

            dataset = {'filenames': list_test_img_dir, "filepath": file_test_path, "classnames": classnames}
            tiled_data = {"size": 256, "overlap": overlap, "unique_area": 0}
            last_activation = last_activations[i] if type(last_activations) is list else last_activations

            test_tiled(model_path=list_CNN_name[i],
                       num_class=list_CNN_num_class[i],
                       save_mask_dir=result_CNN_dir[i] + "_" + str(overlap),
                       last_activation=last_activation,
                       dataset = dataset,
                       tiled_data=tiled_data,
                       )  # , save_dir= "data/split test/")

if __name__ == "__main__":
    test_models_all_dir(str_data="img2img/2023_11_03/",
                        classnames=['output'],
                        list_CNN_num_class=[1,1,1],
                        overlap_list=[128],
                        CNN_name=["model_by_config_tiny_unet_v3",
                                  "model_by_config_tiny_unet_v3_bse",
                                  "model_by_config_tiny_unet_v3_dice"],
                        file_test_path="img2img/data/test")