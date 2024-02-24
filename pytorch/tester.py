from src.test import *
from src.npMetrics import *
import os
import json

################################################ доделать test.py
################################################ разобраться с двойным прогресс баром

def test_models_all_dir(str_data,
                        classnames,
                        list_CNN_num_class,
                        CNN_name,
                        overlap_list,
                        file_test_path,
                        etal_path="segmentation/data/original data/testing",
                        last_activations=None,
                        save_report_path="data/report/",
                        using_metrics=[Jaccard, Dice],
                        only_excel_file=True,
                        use_no_merge_data_for_mertic=False,
                        is_save_result=True):

    list_CNN_name = []
    for name in CNN_name:
        #change_name = "обучение " + str_data + "/" + name + ".pt"
        change_name = str_data + "/" + name + ".pt"
        list_CNN_name.append(change_name)

    if is_save_result:
        result_CNN_dirs = []
        for i in range(len(CNN_name)):
            save_name = "data/result/" + str_data + "/" + str(list_CNN_num_class[i]) + "_class/" + CNN_name[i]
            result_CNN_dirs.append(save_name)
    else:
        result_CNN_dirs = None


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
            last_activation = last_activations[i] if type(last_activations) is list else last_activations

            result_CNN_dir = result_CNN_dirs[i] + "_" + str(overlap) if is_save_result else None

            model_predicts = test_tiled(model_path=list_CNN_name[i],
                                        num_class=list_CNN_num_class[i],
                                        save_mask_dir=result_CNN_dir,
                                        last_activation=last_activation,
                                        dataset = dataset,
                                        tiled_data=tiled_data,
                                       )  # , save_dir= "data/split test/")

            result_metrics_merge,\
            text_result_merge,\
            text_result_merge_all = CalulateMetricsFromModelPredict(model_predicts,
                                                                    CNN_name[i],
                                                                    list_CNN_num_class[i],
                                                                    etal_path=etal_path,
                                                                    class_names=classnames,
                                                                    using_metrics=using_metrics,
                                                                    merge_images=True
                                                                    )

            all_text_results_merge += text_result_merge
            all_text_results_merge_all += text_result_merge_all
            all_results_metrics_merge[CNN_name[i]] = result_metrics_merge

            if use_no_merge_data_for_mertic:
                result_metrics,\
                text_result,\
                text_result_all = CalulateMetricsFromModelPredict(model_predicts,
                                                                  CNN_name[i],
                                                                  list_CNN_num_class[i],
                                                                  etal_path=etal_path,
                                                                  class_names=classnames,
                                                                  using_metrics=using_metrics,
                                                                  merge_images=True
                                                                  )

                all_text_results += text_result
                all_text_results_all += text_result_all
                all_results_metrics[CNN_name[i]] = result_metrics

    #print("str_data before :", str_data)
    if "/" in str_data:
        str_data = str_data.split('/')[-1]
    if '\\' in str_data:
        str_data = str_data.split('\\')[-1]

    save_report_path = os.path.join(save_report_path, str_data)
    # save all reports
    if not os.path.isdir(save_report_path):
        print(f"create dir:'{save_report_path}'")
        os.makedirs(save_report_path)

    #print("str_data is :", str_data)

    if not only_excel_file:
        with open(os.path.join(save_report_path, f'{str_data}_test_models_merge_mean.txt'),'w') as file_mean:
            file_mean.write(all_text_results_merge)

        with open(os.path.join(save_report_path, f'{str_data}_test_models_merge_all.txt'),'w') as file_all:
            file_all.write(all_text_results_merge_all)

        if use_no_merge_data_for_mertic:
            with open(os.path.join(save_report_path, f'{str_data}_test_models_mean.txt'),
                      'w') as file_mean:
                file_mean.write(all_text_results)

            with open(os.path.join(save_report_path, f'{str_data}_test_models_all.txt'),
                      'w') as file_all:
                file_all.write(all_text_results_all)

    test_for_excel_merge = GetFinalTestMetricForExcel(all_results_metrics_merge, using_metrics, classnames)
    with open(os.path.join(save_report_path, f'excel_{str_data}_test_models_merge.txt'),'w') as file_for_excel_merge:
        file_for_excel_merge.write(test_for_excel_merge)

    if use_no_merge_data_for_mertic:
        test_for_excel =       GetFinalTestMetricForExcel(all_results_metrics, using_metrics, classnames)
        with open(os.path.join(save_report_path, f'excel_{str_data}_test_models.txt'), 'w') as file_for_excel:
            file_for_excel.write(test_for_excel)

    print(f"{str_data} test was saved to path '{save_report_path}'")

def test_models_only_all_mito(str_data,
                              classnames,
                              list_CNN_num_class,
                              CNN_name,
                              overlap_list,
                              file_test_path,
                              etal_path,
                              last_activations=None,
                              save_report_path="data/report_mito/",
                              using_metrics=[Jaccard, Dice],
                              only_excel_file=True,
                              use_no_merge_data_for_mertic=False,
                              is_save_result=True):

    list_CNN_name = []
    for name in CNN_name:
        #change_name = "обучение " + str_data + "/" + name + ".pt"
        change_name = str_data + "/" + name + ".pt"
        list_CNN_name.append(change_name)

    if is_save_result:
        result_CNN_dirs = []
        for i in range(len(CNN_name)):
            save_name = "data/result_mito/" + str_data + "/" + str(list_CNN_num_class[i]) + "_class/" + CNN_name[i]
            result_CNN_dirs.append(save_name)
    else:
        result_CNN_dirs = None

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
            last_activation = last_activations[i] if type(last_activations) is list else last_activations

            result_CNN_dir = result_CNN_dirs[i] + "_" + str(overlap) if is_save_result else None

            model_predicts=test_tiled(model_path=list_CNN_name[i],
                                      num_class=list_CNN_num_class[i],
                                      save_mask_dir=result_CNN_dir,
                                      last_activation=last_activation,
                                      dataset = dataset,
                                      tiled_data=tiled_data,
                                      )  # , save_dir= "data/split test/")

            result_metrics_merge,\
            text_result_merge,\
            text_result_merge_all = CalulateMetricsFromModelPredict(model_predicts,
                                                                    CNN_name[i],
                                                                    1,
                                                                    etal_path = etal_path,
                                                                    class_names = classnames,
                                                                    using_metrics=using_metrics,
                                                                    merge_images=True
                                                                   )

            all_text_results_merge += text_result_merge
            all_text_results_merge_all += text_result_merge_all
            all_results_metrics_merge[CNN_name[i]] = result_metrics_merge

            if use_no_merge_data_for_mertic:
                result_metrics,\
                text_result,\
                text_result_all = CalulateMetricsFromModelPredict(model_predicts,
                                                                  CNN_name[i],
                                                                  1,
                                                                  etal_path = etal_path,
                                                                  class_names = classnames,
                                                                  using_metrics=using_metrics,
                                                                  merge_images=False
                                                                  )

                all_text_results += text_result
                all_text_results_all += text_result_all
                all_results_metrics[CNN_name[i]] = result_metrics

    #print("str_data before :", str_data)
    if "/" in str_data:
        str_data = str_data.split('/')[-1]
    if '\\' in str_data:
        str_data = str_data.split('\\')[-1]

    save_report_path = os.path.join(save_report_path, str_data)
    # save all reports
    if not os.path.isdir(save_report_path):
        print(f"create dir:'{save_report_path}'")
        os.makedirs(save_report_path)

    if not only_excel_file:
        with open(os.path.join(save_report_path, f'{str_data}_test_models_merge_mean.txt'),'w') as file_mean:
            file_mean.write(all_text_results_merge)

        with open(os.path.join(save_report_path, f'{str_data}_test_models_merge_all.txt'),'w') as file_all:
            file_all.write(all_text_results_merge_all)

        if use_no_merge_data_for_mertic:
            with open(os.path.join(save_report_path, f'{str_data}_test_models_mean.txt'),
                      'w') as file_mean:
                file_mean.write(all_text_results)

            with open(os.path.join(save_report_path, f'{str_data}_test_models_all.txt'),
                      'w') as file_all:
                file_all.write(all_text_results_all)

    test_for_excel_merge = GetFinalTestMetricForExcel(all_results_metrics_merge, using_metrics, classnames)
    with open(os.path.join(save_report_path, f'excel_{str_data}_test_models_merge.txt'),'w') as file_for_excel_merge:
        file_for_excel_merge.write(test_for_excel_merge)

    if use_no_merge_data_for_mertic:
        test_for_excel =       GetFinalTestMetricForExcel(all_results_metrics, using_metrics, classnames)
        with open(os.path.join(save_report_path, f'excel_{str_data}_test_models.txt'),'w') as file_for_excel:
            file_for_excel.write(test_for_excel)


    print(f"{str_data} test was saved to path '{save_report_path}'")


def test_by_using_config_in_dir(path_to_models, calculate_our_markup=True, calculate_all_mito = True, calculate_all_Lucchipp_mito=True):

    config_file_names = [name for name in os.listdir(path_to_models) if name.endswith(".json") and name.startswith("config_")]
    overlap_list = [128]
    using_metrics = [Dice]
    CNN_names = []
    list_CNN_num_class = []
    last_activations = []

    for config_file_name in config_file_names:
        with open(os.path.join(path_to_models, config_file_name)) as config_buffer:
            config_file = json.load(config_buffer)
        num_classes = config_file["train"]["num_class"]
        list_CNN_num_class.append(num_classes)

        last_activation = config_file["model"]["last_activation"]
        last_activations.append(last_activation)
        ############################################################################################################### Обратная совместимость со старыми файлами
        if "mask_name_label_list" in config_file.keys():
            classnames = config_file["mask_name_label_list"]
        else:
            classnames = config_file["train"]["mask_name_label_list"]

        CNN_name = "model_by_" + config_file_name[:-5]
        CNN_names.append(CNN_name)

    if calculate_our_markup:
        # путь до картинок для теста
        etal_path = "segmentation/data/original data/testing"
        file_test_path = "G:/Data/Unet_multiclass/data/test"
        save_report_path = "data/report/"

        test_models_all_dir(path_to_models,
                            classnames,
                            list_CNN_num_class,
                            CNN_names,
                            overlap_list,
                            file_test_path,
                            last_activations=last_activations,
                            save_report_path=save_report_path,
                            using_metrics=using_metrics)

    if calculate_all_mito:
        # путь до картинок для теста
        file_test_path = "G:/Data/Unet_multiclass/data/orig_EPFL_data/original"
        etal_path = "G:/Data/Unet_multiclass/data/orig_EPFL_data"
        save_report_path = "data/report_epfl_mito/"

        test_models_only_all_mito(path_to_models,
                                  classnames,
                                  list_CNN_num_class,
                                  CNN_names,
                                  overlap_list,
                                  file_test_path,
                                  etal_path,
                                  last_activations=last_activations,
                                  save_report_path=save_report_path,
                                  using_metrics=using_metrics)

    if calculate_all_Lucchipp_mito:
        # путь до картинок для теста
        file_test_path = "G:/Data/Unet_multiclass/data/Luchi_pp_EPFL_test/original"
        etal_path = "G:/Data/Unet_multiclass/data/Luchi_pp_EPFL_test"
        save_report_path = "data/report_lucchipp_mito/"

        test_models_only_all_mito(path_to_models,
                                  classnames,
                                  list_CNN_num_class,
                                  CNN_names,
                                  overlap_list,
                                  file_test_path,
                                  etal_path,
                                  last_activations=last_activations,
                                  save_report_path=save_report_path,
                                  using_metrics = using_metrics)

def datasets_test(calculate_our_markup=True, calculate_all_mito = True, calculate_all_Lucchipp_mito=True):
    using_metrics = [Dice]

    models = [
              #"unet",
              "tiny_unet_v3",
              #"mobile_unet",
              #"Lars76_unet"
             ]

    num_classes = [6]

    types_datasets = [
                      #"real_v2",
                      #"sint_v10",
                      #"mix_v2"
                      "2023_12_20",
                      "2023_12_15"
                      ]

    losses = [
              #"BCELossMulticlass",
              #"MSELossMulticlass",
              "DiceLossMulticlass",
              #"LossDistance2Nearest",
              #"LossDistance2Nearest_DiceLossMulticlass"
              ]

    for type_dataset in types_datasets:
        CNN_names = []
        list_CNN_num_class = []
        path_models = f"segmentation/{type_dataset}"

        for num_class in num_classes:
            for model_name in models:
                for loss in losses:
                    CNN_name = f"model_by_config_proportion_{model_name}"
                    CNN_names.append(CNN_name)
                    list_CNN_num_class.append(num_class)

        overlap_list = [128]
        last_activation = None

        classnames = ["mitochondria", "PSD", "vesicles", "axon", "boundaries", "mitochondrial boundaries"]

        if calculate_our_markup:
            # путь до картинок для теста
            file_test_path = "segmentation/data/original data/testing/original"
            save_report_path = f"data/report/dataset_{type_dataset}"
            etal_path = "segmentation/data/original data/testing/"

            test_models_all_dir(path_models,
                                classnames,
                                list_CNN_num_class,
                                CNN_names,
                                overlap_list,
                                file_test_path,
                                etal_path,
                                last_activations=last_activation,
                                save_report_path=save_report_path,
                                using_metrics=using_metrics)

        if calculate_all_mito:
            # путь до картинок для теста
            file_test_path = "G:/Data/Unet_multiclass/data/orig_EPFL_data/original"
            etal_path = "G:/Data/Unet_multiclass/data/orig_EPFL_data"
            save_report_path = f"data/report_epfl_mito/dataset_{type_dataset}_no_boarder_mode"

            test_models_only_all_mito(path_models,
                                      classnames,
                                      list_CNN_num_class,
                                      CNN_names,
                                      overlap_list,
                                      file_test_path,
                                      etal_path,
                                      last_activations=last_activation,
                                      save_report_path=save_report_path,
                                      using_metrics = using_metrics)

        if calculate_all_Lucchipp_mito:
            # путь до картинок для теста
            file_test_path = "G:/Data/Unet_multiclass/data/Luchi_pp_EPFL_test/original"
            etal_path = "G:/Data/Unet_multiclass/data/Luchi_pp_EPFL_test"
            save_report_path = f"data/report_lucchipp_mito/dataset_{type_dataset}"

            test_models_only_all_mito(path_models,
                                      classnames,
                                      list_CNN_num_class,
                                      CNN_names,
                                      overlap_list,
                                      file_test_path,
                                      etal_path,
                                      last_activations=last_activation,
                                      save_report_path=save_report_path,
                                      using_metrics = using_metrics)

if __name__ == "__main__":
    #datasets_test(True, False, False)
    #datasets_test(False, True, True)

    experiment_path = "segmentation/Multiple_test"

    test_by_using_config_in_dir(experiment_path, True, False, False)