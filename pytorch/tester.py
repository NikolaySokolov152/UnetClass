from src.test import *
from src.npMetrics import *
import os

################################################ сделать с конфиг файлом
################################################ доделать test.py

def test_models_all_dir(str_data,
                        classnames,
                        list_CNN_num_class,
                        CNN_name,
                        overlap_list,
                        file_test_path,
                        save_report_path = "data/report/",
                        using_metrics = [Jaccard, Dice]):

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

    all_text_results_merge = ""
    all_text_results_merge_all = ""
    all_results_metrics_merge = {}
    all_text_results = ""
    all_text_results_all = ""
    all_results_metrics = {}

    for i in range(len(list_CNN_num_class)):
        print(f"predict model '{list_CNN_name[i]}'")
        for overlap in overlap_list:
            print("     predict tiled with overlap: ", overlap)

            dataset = {'filenames': list_test_img_dir, "filepath": file_test_path, "classnames": classnames}
            tiled_data = {"size": 256, "overlap": overlap, "unique_area": 0}

            test_tiled(model_path=list_CNN_name[i],
                       num_class=list_CNN_num_class[i],
                       save_mask_dir=result_CNN_dir[i] + "_" + str(overlap),
                       dataset = dataset,
                       tiled_data=tiled_data,
                       )  # , save_dir= "data/split test/")

            result_metrics_merge,\
            text_result_merge,\
            text_result_merge_all = CalulateMetricsDir(CNN_name[i],
                                                       list_CNN_num_class[i],
                                                       path_train = result_CNN_dir[i] + "_" + str(overlap),
                                                       using_metrics=using_metrics,
                                                       merge_images=True
                                                      )

            all_text_results_merge += text_result_merge
            all_text_results_merge_all += text_result_merge_all
            all_results_metrics_merge[CNN_name[i]] = result_metrics_merge

            result_metrics,\
            text_result,\
            text_result_all = CalulateMetricsDir(CNN_name[i],
                                                 list_CNN_num_class[i],
                                                 path_train = result_CNN_dir[i] + "_" + str(overlap),
                                                 using_metrics=using_metrics,
                                                 merge_images=False
                                                )

            all_text_results += text_result
            all_text_results_all += text_result_all
            all_results_metrics[CNN_name[i]] = result_metrics

    # save all reports
    if not os.path.isdir(save_report_path):
        print(f"create dir:'{save_report_path}'")
        os.makedirs(save_report_path)

    with open(os.path.join(save_report_path, f'{str_data}_test_models_merge_mean.txt'),'w') as file_mean:
        file_mean.write(all_text_results_merge)

    with open(os.path.join(save_report_path, f'{str_data}_test_models_merge_all.txt'),'w') as file_all:
        file_all.write(all_text_results_merge_all)

    with open(os.path.join(save_report_path, f'{str_data}_test_models_mean.txt'),
              'w') as file_mean:
        file_mean.write(all_text_results)

    with open(os.path.join(save_report_path, f'{str_data}_test_models_all.txt'),
              'w') as file_all:
        file_all.write(all_text_results_all)
    print(f"{str_data} test was saved to path '{save_report_path}'")

    test_for_excel_merge = GetFinalTestMetricForExcel(all_results_metrics_merge, using_metrics, classnames)
    test_for_excel =       GetFinalTestMetricForExcel(all_results_metrics, using_metrics, classnames)

    with open(os.path.join(save_report_path, f'excel_{str_data}_test_models_merge.txt'),'w') as file_for_excel_merge:
        file_for_excel_merge.write(test_for_excel_merge)

    with open(os.path.join(save_report_path, f'excel_{str_data}_test_models.txt'),'w') as file_for_excel:
        file_for_excel.write(test_for_excel)


def test_models_only_all_mito(str_data,
                              classnames,
                              list_CNN_num_class,
                              CNN_name,
                              overlap_list,
                              file_test_path,
                              etal_path,
                              save_report_path = "data/report_mito/",
                              using_metrics = [Jaccard, Dice]):

    list_CNN_name = []
    for name in CNN_name:
        #change_name = "обучение " + str_data + "/" + name + ".pt"
        change_name = str_data + "/" + name + ".pt"
        list_CNN_name.append(change_name)

    result_CNN_dir = []

    for i in range(len(CNN_name)):
        save_name = "data/result_mito/" + str_data + "/" + str(list_CNN_num_class[i]) + "_class/" + CNN_name[i]
        result_CNN_dir.append(save_name)

    list_test_img_dir = os.listdir(os.path.join(file_test_path))
    list_test_img_dir = [name for name in list_test_img_dir if name.endswith((".png", ".jpg"))]

    all_text_results_merge = ""
    all_text_results_merge_all = ""
    all_results_metrics_merge = {}
    all_text_results = ""
    all_text_results_all = ""
    all_results_metrics = {}

    for i in range(len(list_CNN_num_class)):
        print(f"predict model '{list_CNN_name[i]}'")
        for overlap in overlap_list:
            print("     predict tiled with overlap: ", overlap)

            dataset = {'filenames': list_test_img_dir, "filepath": file_test_path, "classnames": classnames}
            tiled_data = {"size": 256, "overlap": overlap, "unique_area": 0}

            test_tiled(model_path=list_CNN_name[i],
                       num_class=list_CNN_num_class[i],
                       save_mask_dir=result_CNN_dir[i] + "_" + str(overlap),
                       dataset = dataset,
                       tiled_data=tiled_data,
                       )  # , save_dir= "data/split test/")

            result_metrics_merge,\
            text_result_merge,\
            text_result_merge_all = CalulateMetricsDir(CNN_name[i],
                                                       1,
                                                       etal_path = etal_path,
                                                       path_train = result_CNN_dir[i] + "_" + str(overlap),
                                                       using_metrics=using_metrics,
                                                       class_names = classnames,
                                                       merge_images=True
                                                      )

            all_text_results_merge += text_result_merge
            all_text_results_merge_all += text_result_merge_all
            all_results_metrics_merge[CNN_name[i]] = result_metrics_merge

            result_metrics,\
            text_result,\
            text_result_all = CalulateMetricsDir(CNN_name[i],
                                                 1,
                                                 etal_path = etal_path,
                                                 path_train = result_CNN_dir[i] + "_" + str(overlap),
                                                 using_metrics=using_metrics,
                                                 class_names = classnames,
                                                 merge_images=False
                                                 )

            all_text_results += text_result
            all_text_results_all += text_result_all
            all_results_metrics[CNN_name[i]] = result_metrics

    # save all reports
    if not os.path.isdir(save_report_path):
        print(f"create dir:'{save_report_path}'")
        os.makedirs(save_report_path)

    with open(os.path.join(save_report_path, f'{str_data}_test_models_merge_mean.txt'),'w') as file_mean:
        file_mean.write(all_text_results_merge)

    with open(os.path.join(save_report_path, f'{str_data}_test_models_merge_all.txt'),'w') as file_all:
        file_all.write(all_text_results_merge_all)

    with open(os.path.join(save_report_path, f'{str_data}_test_models_mean.txt'),
              'w') as file_mean:
        file_mean.write(all_text_results)

    with open(os.path.join(save_report_path, f'{str_data}_test_models_all.txt'),
              'w') as file_all:
        file_all.write(all_text_results_all)
    print(f"{str_data} test was saved to path '{save_report_path}'")

    test_for_excel_merge = GetFinalTestMetricForExcel(all_results_metrics_merge, using_metrics, classnames)
    test_for_excel =       GetFinalTestMetricForExcel(all_results_metrics, using_metrics, classnames)

    with open(os.path.join(save_report_path, f'excel_{str_data}_test_models_merge.txt'),'w') as file_for_excel_merge:
        file_for_excel_merge.write(test_for_excel_merge)

    with open(os.path.join(save_report_path, f'excel_{str_data}_test_models.txt'),'w') as file_for_excel:
        file_for_excel.write(test_for_excel)

def standart_test():
    str_data = "Models_and_classes_real_2023_05_10"

    using_metrics = [Dice]

    CNN_names = [
        "model_by_config_real_1_classes_Lars76_unet",
        "model_by_config_real_1_classes_mobile_unet",
        "model_by_config_real_1_classes_tiny_unet",
        "model_by_config_real_1_classes_tiny_unet_v3",
        "model_by_config_real_1_classes_unet",
        "model_by_config_real_5_classes_Lars76_unet",
        "model_by_config_real_5_classes_mobile_unet",
        "model_by_config_real_5_classes_tiny_unet",
        "model_by_config_real_5_classes_tiny_unet_v3",
        "model_by_config_real_5_classes_unet",
        "model_by_config_real_6_classes_Lars76_unet",
        "model_by_config_real_6_classes_mobile_unet",
        "model_by_config_real_6_classes_tiny_unet",
        "model_by_config_real_6_classes_tiny_unet_v3",
        "model_by_config_real_6_classes_unet",
    ]

    types_datasets = ["_real_", "_mix_", "_sint_"]

    last_dataset = "_real_"
    for type_dataset in types_datasets:
        if not type_dataset == "_real_":
            str_data = str_data.replace(last_dataset, type_dataset)
            for i, cnn_name in enumerate(CNN_names):
                CNN_names[i] = cnn_name.replace(last_dataset, type_dataset)

            last_dataset = type_dataset

        list_CNN_num_class = [
            1,
            1,
            1,
            1,
            1,
            5,
            5,
            5,
            5,
            5,
            6,
            6,
            6,
            6,
            6,
        ]
        overlap_list = [128]

        classnames = ["mitochondria", "PSD", "vesicles", "axon", "boundaries", "mitochondrial boundaries"]
        # путь до картинок для теста
        file_test_path = "G:/Data/Unet_multiclass/data/test"

        test_models_all_dir(str_data,
                            classnames,
                            list_CNN_num_class,
                            CNN_names,
                            overlap_list,
                            file_test_path,
                            using_metrics=using_metrics)

        # путь до картинок для теста
        file_test_path = "G:/Data/Unet_multiclass/data/EPFL_test/original"
        etal_path = "G:/Data/Unet_multiclass/data/EPFL_test"

        test_models_only_all_mito(str_data,
                                  classnames,
                                  list_CNN_num_class,
                                  CNN_names,
                                  overlap_list,
                                  file_test_path,
                                  etal_path,
                                  using_metrics = using_metrics)

def activation_test():
    str_data = "Lars_test_2023_05_10"

    using_metrics = [Dice]

    last_activations = ["arctan_activation",
                        "softsign_activation",
                        "sigmoid_activation",
                        "linear_activation",
                        "inv_square_root_activation",
                        "cdf_activation",
                        "hardtanh_activation"]

    types_datasets = ["real", "mix", "sint"]

    losses = ["BCELoss",
              "MSELoss",
              "DiceLoss"]

    for type_dataset in types_datasets:
        CNN_names = []
        list_CNN_num_class = []

        for last_activation in last_activations:
            for loss in losses:
                list_CNN_num_class.append(6)
                CNN_name = "_".join(["model_by_config", type_dataset, last_activation, loss, "Lars76_unet"])
                CNN_names.append(CNN_name)

        print(len(CNN_names))
        overlap_list = [128]
        classnames = ["mitochondria", "PSD", "vesicles", "axon", "boundaries", "mitochondrial boundaries"]

        save_report_path = "data/report_Lars/" + type_dataset
        # путь до картинок для теста
        file_test_path = "G:/Data/Unet_multiclass/data/test"

        test_models_all_dir(str_data,
                            classnames,
                            list_CNN_num_class,
                            CNN_names,
                            overlap_list,
                            file_test_path,
                            save_report_path=save_report_path,
                            using_metrics = using_metrics)

        save_report_path_mito = "data/report_Lars_mito/" + type_dataset
        # путь до картинок для теста
        file_test_path_mito = "G:/Data/Unet_multiclass/data/EPFL_test/original"
        etal_path_mito = "G:/Data/Unet_multiclass/data/EPFL_test"

        #test_models_only_all_mito(str_data,
        #                          classnames,
        #                          list_CNN_num_class,
        #                          CNN_names,
        #                          overlap_list,
        #                          file_test_path_mito,
        #                          etal_path_mito,
        #                          save_report_path=save_report_path_mito,
        #                          using_metrics = using_metrics)

def test_test_config():
    str_data = "2023_05_01"

    CNN_names = [
        "model_by_config_test_tiny_unet_v3_move"
    ]

    list_CNN_num_class = [
        6
    ]
    overlap_list = [128]

    classnames = ["mitochondria", "PSD", "vesicles", "axon", "boundaries", "mitochondrial boundaries"]
    # путь до картинок для теста
    file_test_path = "G:/Data/Unet_multiclass/data/test"

    test_models_all_dir(str_data, classnames, list_CNN_num_class, CNN_names, overlap_list, file_test_path)


if __name__ == "__main__":
    #test_test_config()
    #standart_test()
    activation_test()
