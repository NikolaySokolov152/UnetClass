#import os
import json
import os
import datetime
import time

from src.comparison import CalulateMetricsFromModelPredict, GetFinalTestMetricForExcel
from src.test_metric import METRIC_NAMES
from src.prepare_data import plug_old_connect_res_test_data, extract_channels_by_indexes_from_img_list
from src.test import readPredictDataset, getPipliner, test_data

################# для корректного отображения в пайчарме необходимо включить имитацию консоли

our_marking_test_path = "segmentation/data/original data/testing/"
epfl_marking_test_path = "D:/Data/Unet_multiclass/data/orig_EPFL_data"
lucchipp_marking_test_path = "D:/Data/Unet_multiclass/data/Luchi_pp_EPFL_test"


Kasthuri_test_path = "D:/Data/mito_data/Kasthuri++/"
UroCell_test_path = "D:/Data/mito_data/UroCell-master/"
MouseNucleusAccumbens_test_path = "D:/Data/mito_data/Mouse nucleus accumbens jrc_mus-nacc-1/recon-1/"

batch_test_size = 4

def test_models_all_dir(path_to_model_data,
                        class_check_names,
                        list_CNN_num_class,
                        CNN_config_name_list,
                        overlap_list,
                        test_input_file_path,
                        etal_mask_path="segmentation/data/original data/testing",
                        tiling_mode=True,
                        mask_save_dir_path="data/result/",
                        save_report_path="data/report/",
                        using_metric_names=["Jaccard", "Dice"],
                        only_excel_file=True,
                        use_no_merge_data_for_mertic=False,
                        silence_mode=False,
                        selected_class_indexes=None
                        ):

    all_text_results_merge = ""
    all_text_results_merge_all = ""
    all_results_metrics_merge = []
    all_text_results = ""
    all_text_results_all = ""
    all_results_metrics = []

    dataset_for_predict = readPredictDataset(test_input_file_path, as_gray=True)

    for i in range(len(list_CNN_num_class)):
        all_path_to_model = os.path.join(path_to_model_data, "model_by_" + CNN_config_name_list[i])

        print(f"predict model '{all_path_to_model}'")
        for overlap in overlap_list:
            print("     predict tiled with overlap: ", overlap)

            model, name_model = getPipliner(path_to_model_data, CNN_config_name_list[i])
            model.silence_mode = silence_mode
            #model.silence_mode = True

            if mask_save_dir_path is None:
                result_CNN_dir = None
            else:
                if os.path.isabs(path_to_model_data):
                    drive, work_path_to_model_data = os.path.splitdrive(path_to_model_data)
                else:
                    work_path_to_model_data = path_to_model_data

                result_CNN_dir = os.path.join(mask_save_dir_path,
                                              work_path_to_model_data[1:],
                                              f"{list_CNN_num_class[i]}_class",
                                              name_model) + "_" + str(overlap)

            tiled_data = {"size": 256, "overlap": overlap, "unique_area": 0} if tiling_mode else None

            predict_img_list, predict_name_list = test_data(model,
                                                            dataset_for_predict,
                                                            save_mask_dir=result_CNN_dir,
                                                            tiled_data=tiled_data,
                                                            batch_size=batch_test_size,
                                                            # save_spliting_dir="data/split test/"
                                                            )

            if model.num_classes < (list_CNN_num_class[i] if selected_class_indexes is None else max(selected_class_indexes)+1):
                if selected_class_indexes is None:
                    num_check_classes = model.num_classes
                else:
                    change_selected_class_indexes = []
                    for index in selected_class_indexes:
                        if index < model.num_classes:
                            change_selected_class_indexes.append(index)
                    selected_class_indexes = change_selected_class_indexes
                    num_check_classes = len(selected_class_indexes)

                print(f"Warning! The model contains fewer classes than used in testing! I use the maximum number of classes {num_check_classes} out of possible.")
            else:
                num_check_classes = list_CNN_num_class[i] if selected_class_indexes is None else len(selected_class_indexes)

            model_predicts_list = predict_img_list if selected_class_indexes is None else extract_channels_by_indexes_from_img_list(predict_img_list, selected_class_indexes)

            predict_for_check = plug_old_connect_res_test_data(model_predicts_list, predict_name_list)

            if use_no_merge_data_for_mertic:
                result_metrics,\
                text_result,\
                text_result_all = CalulateMetricsFromModelPredict(predict_for_check,
                                                                  name_model,
                                                                  num_check_classes,
                                                                  etal_path=etal_mask_path,
                                                                  class_names=class_check_names,
                                                                  using_metric_names=using_metric_names,
                                                                  merge_images=False,
                                                                  is_print_metric=False
                                                                  )

                all_text_results += text_result
                all_text_results_all += text_result_all
                all_results_metrics.append(result_metrics)

            result_metrics_merge,\
            text_result_merge,\
            text_result_merge_all = CalulateMetricsFromModelPredict(predict_for_check,
                                                                    name_model,
                                                                    num_check_classes,
                                                                    etal_path=etal_mask_path,
                                                                    class_names=class_check_names,
                                                                    using_metric_names=using_metric_names,
                                                                    merge_images=True,
                                                                    is_print_metric=False
                                                                    )

            all_text_results_merge += text_result_merge
            all_text_results_merge_all += text_result_merge_all
            all_results_metrics_merge.append(result_metrics_merge)

    #print("str_data before :", str_data)
    if "/" in path_to_model_data:
        str_data = path_to_model_data.split('/')[-1]
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

        test_for_excel_merge = GetFinalTestMetricForExcel(all_results_metrics_merge, using_metric_names, class_check_names)
        with open(os.path.join(save_report_path, f'excel_{str_data}_test_models_merge.csv'),'w') as file_for_excel_merge:
            file_for_excel_merge.write(test_for_excel_merge)

        if use_no_merge_data_for_mertic:
            test_for_excel =   GetFinalTestMetricForExcel(all_results_metrics, using_metric_names, class_check_names)
            with open(os.path.join(save_report_path, f'excel_{str_data}_test_models.csv'), 'w') as file_for_excel:
                file_for_excel.write(test_for_excel)

        print(f"{str_data} test was saved to path '{save_report_path}'")

    return (all_results_metrics_merge,
            all_results_metrics if use_no_merge_data_for_mertic else None,
            using_metric_names,
            class_check_names)

def sort_list_by_name_list(path_list, name_list):
    res_list = []
    for search_name in name_list:
        res_list += [name for name in path_list if search_name in name]
        print(f"sorting by {search_name}, len of list {len(res_list)}")
    return res_list


def sorting_names(names_list):
    names_list = sort_list_by_name_list(names_list, ["1_classes_dataset_",
                                                                   "5_classes_dataset_",
                                                                   "6_classes_dataset_"])
    names_list = sort_list_by_name_list(names_list, ["mix_", "diff_"])
    return names_list
def test_by_using_config_in_dir(path_to_models, list_of_description_test_dataset):

    if isinstance(path_to_models, str) and path_to_models.endswith(".json"):
        path_to_dir, file_name = os.path.split(path_to_models)
        config_file_names = [file_name]
        work_path = path_to_dir
    elif isinstance(path_to_models, list) and path_to_models[0].endswith(".json"):
        path_to_dir = os.path.commonpath(path_to_models)
        config_file_names = [name.replace(path_to_dir, "") for name in path_to_models]
        work_path = path_to_dir
    else:
        config_file_names = [name for name in os.listdir(path_to_models) if name.endswith(".json") and name.startswith("config_")]
        work_path = path_to_models

    overlap_list = [128]
    using_metric_names = ["Dice"]
    CNN_config_name_list = []
    list_CNN_num_class = []

    for name in config_file_names:
        print(name)

    classnames = None

    for config_file_name in config_file_names:
        with open(os.path.join(work_path, config_file_name)) as config_buffer:
            config_file = json.load(config_buffer)
        num_classes = config_file["train"]["num_class"]
        list_CNN_num_class.append(num_classes)

        ############################################################################################################### Обратная совместимость со старыми файлами
        if "mask_name_label_list" in config_file.keys():
            model_classnames = config_file["mask_name_label_list"]
        else:
            model_classnames = config_file["train"]["mask_name_label_list"]

        if classnames is None:
            classnames = model_classnames
        else:
            for class_name in model_classnames:
                if not class_name in classnames:
                    print(f'WARNING!!! Additional class {class_name} found!!! All list is {len(model_classnames)} classes with name "{", ".join(classnames)}"!')
                model_classnames.append(class_name)

        CNN_config_name_list.append(config_file_name[:-5])

    result = []
    for deck_of_dataset in list_of_description_test_dataset:
        test_input_file_path = deck_of_dataset["test_input_file_path"]
        etal_mask_path = deck_of_dataset["etal_mask_path"]
        class_names_list = deck_of_dataset["class_names_list"] if "class_names_list" in deck_of_dataset.keys() else classnames
        dataset_name = deck_of_dataset["dataset_name"]
        #save_report_path = deck_of_dataset["save_report_path"]
        save_split_report_path = deck_of_dataset["save_split_report_path"] if "save_split_report_path" in deck_of_dataset.keys() else None
        mask_save_dir_path = deck_of_dataset["mask_save_dir_path"] if "mask_save_dir_path" in deck_of_dataset.keys() else "data/result"
        selected_class_indexes = deck_of_dataset["selected_class_indexes"] if "selected_class_indexes" in deck_of_dataset.keys() else None

        dataset_result=test_models_all_dir(work_path,
                                           class_names_list,
                                           list_CNN_num_class,
                                           CNN_config_name_list,
                                           overlap_list,
                                           test_input_file_path=test_input_file_path,
                                           etal_mask_path=etal_mask_path,
                                           save_report_path=save_split_report_path,
                                           using_metric_names=using_metric_names,
                                           mask_save_dir_path=mask_save_dir_path,
                                           selected_class_indexes=selected_class_indexes)

        result.append((dataset_name, dataset_result))

    return result

def checking_models_list_on_one_dataset (str_data,
                                         experiment_paths,
                                         list_of_description_test_dataset,
                                         save_report_path="data/report/"):

    all_results_metrics_merge=[]
    all_results_metrics=[]
    using_metrics=None
    classnames=None

    for experiment_path in experiment_paths:
       dirs_result = test_by_using_config_in_dir(experiment_path,
                                                 list_of_description_test_dataset
                                                 )

       name_dataset, result = dirs_result[0]

       one_our, two_our, tree_our, four_our = result

       all_results_metrics_merge += one_our
       if two_our is not None:
           all_results_metrics += two_our
       using_metrics = tree_our
       classnames = four_our

    save_report_path = os.path.join(save_report_path, str_data)
    if not os.path.isdir(save_report_path):
       print(f"create dir:'{save_report_path}'")
       os.makedirs(save_report_path)

    test_for_excel_merge = GetFinalTestMetricForExcel(all_results_metrics_merge, using_metrics, classnames)
    with open(os.path.join(save_report_path, f'excel_{str_data}_test_models_merge.csv'),'w') as file_for_excel_merge:
       file_for_excel_merge.write(test_for_excel_merge)

    return test_for_excel_merge

def RunMultiTestsSeriesExpOnMultiDatasets(str_data, experiment_paths, save_mask=True):
    save_report_all_dataset_path = "data/all_report/"
    save_report_path = None
    list_of_description_test_dataset = [
        {"dataset_name": "UnnTestDataset",
         "test_input_file_path": os.path.join(our_marking_test_path, "original"),
         "etal_mask_path": our_marking_test_path,
         #"class_names_list":,
         "save_report_path": "data/report/",
         "mask_save_dir_path": "data/result/" if save_mask else None
         },

        {"dataset_name": "LucchiPPTestDataset",
         "test_input_file_path": os.path.join(lucchipp_marking_test_path, "original"),
         "etal_mask_path": lucchipp_marking_test_path,
         "class_names_list": ["mitochondria"],
         "save_report_path": "data/report_lucchipp_mito/",
         "mask_save_dir_path": "data/result_mito/"  if save_mask else None,
         "selected_class_indexes": [0]
         },

        #{"dataset_name": "EPFLTestDataset",
        # "test_input_file_path": os.path.join(epfl_marking_test_path, "original"),
        # "etal_mask_path": epfl_marking_test_path,
        # # "class_names_list":,
        # "save_report_path": "data/report_epfl_mito/",
        # "mask_save_dir_path": None,
        # "selected_class_indexes": [0]
        # },

        {"dataset_name": "KasthuriTestDataset",
         "test_input_file_path": os.path.join(Kasthuri_test_path, "Test_In"),
         "etal_mask_path": Kasthuri_test_path,
         "class_names_list": ["Test_Out"],
         "save_report_path": "data/report_kasthuri_mito/",
         "mask_save_dir_path": "data/result_kasthuri_mito/" if save_mask else None,
         "selected_class_indexes": [0]
         },

        {"dataset_name": "UroCellDataset",
         "test_input_file_path": os.path.join(UroCell_test_path, "data/fib1-0-0-0"),
         "etal_mask_path": UroCell_test_path,
         "class_names_list": ["mito/binary/fib1-0-0-0"],
         "save_report_path": "data/report_urocell_mito/",
         "mask_save_dir_path": "data/result_urocell_mito/" if save_mask else None,
         "selected_class_indexes": [0]
         },

        {"dataset_name": "MouseNucleusAccumbens",
         "test_input_file_path": os.path.join(MouseNucleusAccumbens_test_path, "em/fibsem-uint8"),
         "etal_mask_path": MouseNucleusAccumbens_test_path,
         "class_names_list": ["labels/groundtruth/crop115/mito", "labels/groundtruth/crop115/ves", "labels/groundtruth/crop115/pm", "labels/groundtruth/crop115/mito_mem"],
         "save_report_path": "data/report_mna/",
         "mask_save_dir_path": "data/result_mna/" if save_mask else None,
         "selected_class_indexes": [0, 2, 4, 5]
         },
    ]

    #list_of_description_test_dataset = list_of_description_test_dataset[:-1]
    #list_of_description_test_dataset = [list_of_description_test_dataset[-1]]

    datasets_res = []
    for dataset in list_of_description_test_dataset:
        print(f'Test on dataset "{dataset["dataset_name"]}"')
        test_for_excel_merge = checking_models_list_on_one_dataset(str_data,
                                                                   experiment_paths,
                                                                   [dataset],
                                                                   save_report_path=dataset["save_report_path"])
        datasets_res.append((test_for_excel_merge, dataset["dataset_name"]))


    all_text_excel = ""
    for text_res, name_dataset in datasets_res:
        all_text_excel += name_dataset + "\n"
        all_text_excel += text_res + "\n"

    if not os.path.isdir(save_report_all_dataset_path):
        print(f"create dir:'{save_report_all_dataset_path}'")
        os.makedirs(save_report_all_dataset_path)
    with open(os.path.join(save_report_all_dataset_path, f'excel_{str_data}_test_models_merge.csv'),'w') as file_for_excel_merge:
       file_for_excel_merge.write(all_text_excel)

def file_log_parcer(file):
    pars_list = []
    model_path_list = []

    for line in file.readlines():
        if len(line.split(' - ')) >= 4:
            d = dict()
            d['date'] = line.split(' - ')[0]
            d['type'] = line.split(' - ')[2]
            message = line.split(' - ')[3]
            d["model_name"] = message.split('"')[1]
            #d['message'] = message
            d["type_with_data_save"] = message.split('"')[3]
            model_path_list.append(os.path.join(d["type_with_data_save"], d["model_name"]))
            pars_list.append(d)
    return pars_list, model_path_list

def runExperimentByLogs(series_experiment_name=None, log_path="experiments_log.log"):
    if not os.path.isfile(log_path):
        print("Experiment file not found!")
    else:
        if series_experiment_name is None:
           now = datetime.datetime.now()
           series_experiment_name = f"{now.year:04}_{now.month:02}_{now.day:02}_{now.hour:02}_{now.minute:02}_{now.second:02}"
        
        with open(log_path, 'r') as f:
                _, model_path_list = file_log_parcer(f)

        RunMultiTestsSeriesExpOnMultiDatasets(series_experiment_name, model_path_list)
        os.remove(log_path)

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

    #experiment_paths = [f"segmentation/Multiple_segmentation_stability_ballancy_and_dataset_100",
    #                    f"segmentation/Multiple_segmentation_stability_ballancy_and_dataset_200",]

    #str_data = "10_12_2024_balance_experiment"
    #main(str_data, experiment_paths)

    #experiment_paths = [f"segmentation/2000_images"]

    #_half_diff

    #experiment_paths = [f"segmentation/Multiple_segmentation_stability_100",
    #                    f"segmentation/Multiple_segmentation_stability_165",]

    '''
    experiment_paths = []

    n_slices = [5, 10, 15] #, 20, 30, 42]
    n_classes = [1, 5, 6]

    for n_slice in n_slices:
        for n_class in n_classes:
            experiment_paths.append(f"segmentation/Multiple_segmentation_stability_{n_slice}_{n_class}_classes_half_diff")


    experiment_paths = sorting_names(experiment_paths)

    str_data = "re_re_experiment_2k_diff_data_half_diff"
    RunMultiTestsSeriesExpOnMultiDatasets(str_data, experiment_paths)
    '''


    RunMultiTestsSeriesExpOnMultiDatasets("test_working", ["segmentation/2025_08_26"])
    #runExperimentByLogs()