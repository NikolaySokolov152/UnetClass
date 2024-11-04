from src.test import *
from src.npMetrics import *
import os
import json

################################################ доделать test.py
################################################ разобраться с двойным прогресс баром

our_marking_test_path = "segmentation/data/original data/testing/"
epfl_marking_test_path = "D:/Data/Unet_multiclass/data/orig_EPFL_data"
lucchipp_marking_test_path = "D:/Data/Unet_multiclass/data/Luchi_pp_EPFL_test"

def test_models_all_dir(str_data,
                        classnames,
                        list_CNN_num_class,
                        CNN_name,
                        overlap_list,
                        file_test_path,
                        etal_path="segmentation/data/original data/testing",
                        last_activations=None,
                        save_dir_path="data/result/",
                        save_report_path="data/report/",
                        using_metrics=[Jaccard, Dice],
                        only_excel_file=True,
                        use_no_merge_data_for_mertic=False,
                        test_only_first_class=False):

    list_CNN_name = []
    for name in CNN_name:
        #change_name = "обучение " + str_data + "/" + name + ".pt"
        change_name = str_data + "/" + name + ".pt"
        list_CNN_name.append(change_name)

    if save_dir_path is None:
        result_CNN_dirs = None
    else:
        result_CNN_dirs = []
        for i in range(len(CNN_name)):
            save_name = os.path.join(save_dir_path, str_data, f"{list_CNN_num_class[i]}_class", CNN_name[i])
            result_CNN_dirs.append(save_name)

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

            result_CNN_dir = None if save_dir_path is None else result_CNN_dirs[i] + "_" + str(overlap)

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
                                                                    list_CNN_num_class[i] if not test_only_first_class else 1,
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
                                                                  list_CNN_num_class[i] if not test_only_first_class else 1,
                                                                  etal_path=etal_path,
                                                                  class_names=classnames,
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

    if save_report_path is not None:
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
            test_for_excel =   GetFinalTestMetricForExcel(all_results_metrics, using_metrics, classnames)
            with open(os.path.join(save_report_path, f'excel_{str_data}_test_models.txt'), 'w') as file_for_excel:
                file_for_excel.write(test_for_excel)

        print(f"{str_data} test was saved to path '{save_report_path}'")

    return all_results_metrics_merge, all_results_metrics if use_no_merge_data_for_mertic else None, using_metrics, classnames

# Заменяет некоторые значения по умолчанию основной функции test_models_all_dir и поднимает флаг test_only_first_class
def test_models_only_all_mito(str_data,
                              classnames,
                              list_CNN_num_class,
                              CNN_name,
                              overlap_list,
                              file_test_path,
                              etal_path,
                              last_activations=None,
                              save_dir_path="data/result_mito/",
                              save_report_path="data/report_mito/",
                              using_metrics=[Jaccard, Dice],
                              only_excel_file=True,
                              use_no_merge_data_for_mertic=False):

    return test_models_all_dir(str_data=str_data,
                               classnames=classnames,
                               list_CNN_num_class=list_CNN_num_class,
                               CNN_name=CNN_name,
                               overlap_list=overlap_list,
                               file_test_path=file_test_path,
                               etal_path=etal_path,
                               last_activations=last_activations,
                               save_dir_path=save_dir_path,
                               save_report_path=save_report_path,
                               using_metrics=using_metrics,
                               only_excel_file=only_excel_file,
                               use_no_merge_data_for_mertic=use_no_merge_data_for_mertic,
                               test_only_first_class=True)

def test_by_using_config_in_dir(path_to_models, calculate_our_markup=True, calculate_all_mito = True, calculate_all_Lucchipp_mito=True, save_mask="data/result"):

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

    our_result=None
    epfl_result=None
    lucchipp_result=None

    if calculate_our_markup:
        # путь до картинок для теста
        etal_path = our_marking_test_path
        file_test_path = os.path.join(etal_path, "original")
        #save_report_path = "data/report/"
        save_report_path = None

        our_result=test_models_all_dir(path_to_models,
                                       classnames,
                                       list_CNN_num_class,
                                       CNN_names,
                                       overlap_list,
                                       file_test_path,
                                       etal_path=etal_path,
                                       last_activations=last_activations,
                                       save_report_path=save_report_path,
                                       using_metrics=using_metrics,
                                       save_dir_path=save_mask)

    if calculate_all_mito:
        # путь до картинок для теста
        etal_path = epfl_marking_test_path
        file_test_path = os.path.join(etal_path, "original")
        #save_report_path = "data/report_epfl_mito/"
        save_report_path = None

        epfl_result=test_models_only_all_mito(path_to_models,
                                              classnames,
                                              list_CNN_num_class,
                                              CNN_names,
                                              overlap_list,
                                              file_test_path,
                                              etal_path,
                                              last_activations=last_activations,
                                              save_report_path=save_report_path,
                                              using_metrics=using_metrics,
                                              save_dir_path=save_mask+"_mito" if save_mask is not None else save_mask)

    if calculate_all_Lucchipp_mito:
        # путь до картинок для теста
        etal_path = lucchipp_marking_test_path
        file_test_path = os.path.join(etal_path, "original")
        #save_report_path = "data/report_lucchipp_mito/"
        save_report_path = None

        lucchipp_result=test_models_only_all_mito(path_to_models,
                                                  classnames,
                                                  list_CNN_num_class,
                                                  CNN_names,
                                                  overlap_list,
                                                  file_test_path,
                                                  etal_path,
                                                  last_activations=last_activations,
                                                  save_report_path=save_report_path,
                                                  using_metrics=using_metrics,
                                                  save_dir_path=save_mask)

    return our_result, epfl_result, lucchipp_result

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
            etal_path = our_marking_test_path
            file_test_path = os.path.join(etal_path, "original")

            save_report_path = f"data/report/dataset_{type_dataset}"

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
            etal_path = epfl_marking_test_path
            file_test_path = os.path.join(etal_path, "original")

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
            etal_path = lucchipp_marking_test_path
            file_test_path = os.path.join(etal_path, "original")

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

def main(str_data, experiment_paths):

    all_results_metrics_merge_our={}
    all_results_metrics_our={}
    using_metrics_our=None
    classnames_our=None

    for experiment_path in experiment_paths:
       our, epfl, lucchipp = test_by_using_config_in_dir(experiment_path, True, False, False)
       one_our, two_our, tree_our, four_our = our

       all_results_metrics_merge_our |= one_our
       if two_our is not None:
           all_results_metrics_our |= two_our
       using_metrics_our = tree_our
       classnames_our = four_our

    save_report_path = "data/report/"
    save_report_path = os.path.join(save_report_path, str_data)
    if not os.path.isdir(save_report_path):
       print(f"create dir:'{save_report_path}'")
       os.makedirs(save_report_path)

    test_for_excel_merge = GetFinalTestMetricForExcel(all_results_metrics_merge_our, using_metrics_our, classnames_our)
    with open(os.path.join(save_report_path, f'excel_{str_data}_test_models_merge.txt'),'w') as file_for_excel_merge:
       file_for_excel_merge.write(test_for_excel_merge)

    all_results_metrics_merge_epfl={}
    all_results_metrics_epfl={}
    using_metrics_epfl=None
    classnames_epfl=None

    for experiment_path in experiment_paths:
       our, epfl, lucchipp = test_by_using_config_in_dir(experiment_path, False, True, False)
       one_epfl, two_epfl, tree_epfl, four_epfl = epfl

       all_results_metrics_merge_epfl |= one_epfl
       if two_epfl is not None:
           all_results_metrics_epfl |= two_epfl
       using_metrics_epfl = tree_epfl
       classnames_epfl = four_epfl

    save_report_path = "data/report_epfl_mito/"
    save_report_path = os.path.join(save_report_path, str_data)
    if not os.path.isdir(save_report_path):
       print(f"create dir:'{save_report_path}'")
       os.makedirs(save_report_path)

    test_for_excel_merge = GetFinalTestMetricForExcel(all_results_metrics_merge_epfl, using_metrics_epfl, classnames_epfl)
    with open(os.path.join(save_report_path, f'excel_{str_data}_test_models_merge.txt'), 'w') as file_for_excel_merge:
       file_for_excel_merge.write(test_for_excel_merge)

    all_results_metrics_merge_lucchi={}
    all_results_metrics_lucchi={}
    using_metrics_lucchi=None
    classnames_lucchi=None

    for experiment_path in experiment_paths:
        our, epfl, lucchipp =test_by_using_config_in_dir(experiment_path, False, False, True, save_mask=None)
        one_lucchi, two_lucchi, tree_lucchi, four_lucchi = lucchipp

        all_results_metrics_merge_lucchi |= one_lucchi
        if two_lucchi is not None:
            all_results_metrics_lucchi |= two_lucchi
        using_metrics_lucchi = tree_lucchi
        classnames_lucchi = four_lucchi

    save_report_path = "data/report_lucchipp_mito/"
    save_report_path = os.path.join(save_report_path, str_data)
    if not os.path.isdir(save_report_path):
        print(f"create dir:'{save_report_path}'")
        os.makedirs(save_report_path)

    test_for_excel_merge = GetFinalTestMetricForExcel(all_results_metrics_merge_lucchi, using_metrics_lucchi, classnames_lucchi)
    with open(os.path.join(save_report_path, f'excel_{str_data}_test_models_merge.txt'), 'w') as file_for_excel_merge:
        file_for_excel_merge.write(test_for_excel_merge)

if __name__ == "__main__":

    """
    type_datasets = ["", "_only_dif", "_only_real"]

    experiment_paths = []
    for dataset in type_datasets:
            experiment_paths.append(f"segmentation/Multiple_synt_and_diffusion_6_classes{dataset}")

    str_data = "our_diffusion_slices_experiment_sd"
    main(str_data, experiment_paths)

    experiment_paths_no_augm = []
    for dataset in type_datasets:
            experiment_paths_no_augm.append(
                    f"segmentation/Multiple_synt_and_diffusion_6_classes{dataset}_no_augment")

    str_data_no_augm = "our_diffusion_slices_experiment_sd_no_augment"
    main(str_data_no_augm, experiment_paths_no_augm)

    n_slices = [5, 10, 15, 20, 30, 42]
    n_classes= [1, 5, 6]
    type_datasets = ["", "_only_dif", "_only_real"]

    experiment_paths = []
    for dataset in type_datasets:
        for n_slice in n_slices:
            for n_class in n_classes:
                experiment_paths.append(f"segmentation/Multiple_diffusion_{n_slice}_slices_{n_class}_classes{dataset}")

    str_data = "our_diffusion_slices_experiment"
    main(str_data, experiment_paths)

    experiment_paths_no_augm = []
    for dataset in type_datasets:
        for n_slice in n_slices:
            for n_class in n_classes:
                experiment_paths_no_augm.append(
                    f"segmentation/Multiple_diffusion_{n_slice}_slices_{n_class}_classes{dataset}_no_augment")

    str_data_no_augm = "our_diffusion_slices_experiment_no_augment"
    main(str_data_no_augm, experiment_paths_no_augm)

    experiment_paths = [f"segmentation/diffusion_100_slices_1_classes",
                        f"segmentation/diffusion_165_slices_1_classes"]
    str_data = "our_add_experiment"
    main(str_data, experiment_paths)
    
    """

    '''
    n_slices = [5, 10, 15, 20, 30, 42]
    n_classes= [1, 5, 6]

    experiment_paths = []

    for n_slice in n_slices:
        for n_class in n_classes:
            experiment_paths.append(f"segmentation/Multiple_diffusion_{n_slice}_slices_{n_class}_classes")

    str_data = "our_diffusion_slices_stability_experiment"
    main(str_data, experiment_paths)

    experiment_paths = [f"segmentation/Multiple_diffusion_100_slices_1_classes",
                        f"segmentation/Multiple_diffusion_165_slices_1_classes"]
    str_data = "our_add_stability_experiment"
    main(str_data, experiment_paths)
    '''

    experiment_paths = [f"segmentation/Multiple_all_datasets_and_synt_and_diffusion_6_classes_stability",
                        f"segmentation/Multiple_diffusion_100_slices_1_classes_mod",
                        f"segmentation/Multiple_diffusion_165_slices_1_classes_mod",
                        f"segmentation/Multiple_only_synt_6_classes_stability",
                        f"segmentation/Multiple_synt_and_diffusion_6_classes_stability"]

    str_data = "18_06_2024_our_stability_experiment"
    main(str_data, experiment_paths)