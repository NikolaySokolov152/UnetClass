import sys

if not __name__ == "__main__":
    sys.path.append("src/")

import json
import numpy as np
import os
import cv2

from test_metric import (METRIC_NAMES, METRIC_FUN)

############################################### передлелать структуру, вынеся сами метрики в отдельный файл
############################################### сделать сохранение в таблицу пандаса
from read_data_transform_fun import to_0_255_format_img


def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def calculateMetrics(y_true, y_pred, using_metrics = []):
    res = {}
    for metric, names in using_metrics:
        vals = metric(y_true, y_pred)
        if not metric.__name__ == "CrowdsourcingMetrics":
            vals = [vals]
        for i, name in enumerate(names):
            res[name] = vals[i]
    return res

# Вычисляет все данные метрики для каждого класса одного реального изображения с передачей предсказания модели
def EvaluateSingleImageModelResultsFromPredict(etal_path,
                                               model_predict_val,
                                               num_classes,
                                               class_names,
                                               using_metrics,
                                               threshold=128):

    img_name = model_predict_val[0]
    model_output = model_predict_val[1]

    res = {"test_img_name":img_name}
    # cycle through classes
    for i in range(num_classes):
        class_name = class_names[i]
        predict_img = model_output.take(i, axis=-1)

        etal_img_path = os.path.join(etal_path, class_name, img_name)
        etal = cv2.imread(etal_img_path, cv2.IMREAD_GRAYSCALE)
        etal = to_0_255_format_img(etal)
        if (etal is None):
            print("error etal")

        pred_img = to_0_255_format_img(predict_img)
        if (pred_img is None):
            print("error predict img")

        # бинаризация с порогом (на всякий случай)
        bin_true = etal.copy()
        bin_true[bin_true<threshold]  =0
        bin_true[bin_true>threshold-1]=255

        bin_img_true = pred_img.copy()
        bin_img_true[bin_img_true<threshold]  =0
        bin_img_true[bin_img_true>threshold-1]=255

        # c векторами работать легче и нет требований на работу с окрестностями пикселей
        y_true = bin_true.ravel()
        y_pred = bin_img_true.ravel()

        res[class_name.replace(' ', '_')]=calculateMetrics(y_true, y_pred, using_metrics)
    return res

# Вычисляет все данные метрики для каждого класса одного реального изображения с передачей предсказания модели
def EvaluateSingleImageModelResultsFromPredictAndEtal(etal_data,
                                                      name_data,
                                                      model_predict_val,
                                                      num_classes,
                                                      class_names,
                                                      using_metrics,
                                                      threshold=128
                                                      ):

    img_name = name_data
    model_output = model_predict_val

    res = {"test_img_name":img_name}
    # cycle through classes
    for i in range(num_classes):
        class_name = class_names[i]
        predict_img = model_output.take(i, axis=-1)
        etal = to_0_255_format_img(etal_data.take(i, axis=-1))
        if (etal is None):
            print("error etal")

        pred_img = to_0_255_format_img(predict_img)
        if (pred_img is None):
            print("error predict img")

        # бинаризация с порогом (на всякий случай)
        bin_true = etal.copy()
        bin_true[bin_true<threshold]  =0
        bin_true[bin_true>threshold-1]=255

        bin_img_true = pred_img.copy()
        bin_img_true[bin_img_true<threshold]  =0
        bin_img_true[bin_img_true>threshold-1]=255

        # c векторами работать легче и нет требований на работу с окрестностями пикселей
        y_true = bin_true.ravel()
        y_pred = bin_img_true.ravel()

        res[class_name.replace(' ', '_')]=calculateMetrics(y_true, y_pred, using_metrics)
    return res


# Вычисляет все данные метрики для каждого класса одного реального изображения
def EvaluateSingleImageModelResults(etal_path,
                                    predict_path,
                                    test_img_name,
                                    predict_prefix,
                                    num_classes,
                                    class_names,
                                    using_metrics,
                                    threshold=128):
    predict_data = []
    # cycle through classes
    for i in range(num_classes):
        class_name = class_names[i]

        predict_img_path = os.path.join(predict_path, class_name, predict_prefix+test_img_name)
        pred_img = cv2.imread(predict_img_path, cv2.IMREAD_GRAYSCALE)
        if (pred_img is None):
            print("error predict img")
        predict_data.append(pred_img)
    # переместить каналлы в конец
    predict=np.array(pred_img).transpose(np.roll(np.range(pred_img.ndim), -1))
    one_model_predict=(test_img_name, predict)

    # отправить в функцию
    return EvaluateSingleImageModelResultsFromPredict(etal_path,
                                                      one_model_predict,
                                                      num_classes,
                                                      class_names,
                                                      using_metrics,
                                                      threshold=threshold)

# Вычисляет все данные метрики для каждого класса со всеми эталонами сразу с передачей предсказания модели
def EvaluateMergeImageModelResultsFromPredict(etal_path,
                                              model_predicts,
                                              num_classes,
                                              class_names,
                                              using_metrics,
                                              threshold=128):
    res = {"test_img_name": f"combined_prediction_by_{len(model_predicts)}_images"}
    # cycle through classes
    for i in range(num_classes):
        class_name = class_names[i]

        etalons_merge = []
        pred_imgs_merge = []

        for test_img_name, pred_img in model_predicts:
            etal_img_path = os.path.join(etal_path, class_name, test_img_name)
            etal = cv2.imread(etal_img_path, cv2.IMREAD_GRAYSCALE)
            if (etal is None):
                raise Exception(f'error etalon img with path {etal_img_path}')
            etal = to_0_255_format_img(etal)

            # обработка предикта на всякий случай
            pred_img = to_0_255_format_img(pred_img.take(i, axis=-1))
            # бинаризация с порогом (на всякий случай)
            bin_true = etal.copy()
            bin_true[bin_true < threshold] = 0
            bin_true[bin_true > threshold - 1] = 255

            bin_pred = pred_img.copy()
            bin_pred[bin_pred < threshold] = 0
            bin_pred[bin_pred > threshold - 1] = 255

            etalons_merge.append(bin_true)
            pred_imgs_merge.append(bin_pred)

        # c векторами работать легче и нет требований на работу с окрестностями пикселей
        y_true = np.array(etalons_merge).ravel()
        y_pred = np.array(pred_imgs_merge).ravel()

        res[class_name.replace(' ', '_')] = calculateMetrics(y_true, y_pred, using_metrics)
    return res

def EvaluateMergeImageModelResultsFromPredictAndEtal(etal_data,
                                                     model_predicts,
                                                     num_classes,
                                                     class_names,
                                                     using_metrics,
                                                     threshold=128):
    res = {"test_img_name": f"combined_prediction_by_{len(model_predicts)}_images"}
    # cycle through classes
    for i in range(num_classes):
        class_name = class_names[i]

        etalons_merge = []
        pred_imgs_merge = []

        check_data = zip(etal_data, model_predicts)

        for etal, pred_img in check_data:
            etal = to_0_255_format_img(etal.take(i, axis=-1))
            # обработка предикта на всякий случай
            pred_img = to_0_255_format_img(pred_img.take(i, axis=-1))
            # бинаризация с порогом (на всякий случай)
            bin_true = etal.copy()
            bin_true[bin_true < threshold] = 0
            bin_true[bin_true > threshold - 1] = 255

            bin_pred = pred_img.copy()
            bin_pred[bin_pred < threshold] = 0
            bin_pred[bin_pred > threshold - 1] = 255

            etalons_merge.append(bin_true)
            pred_imgs_merge.append(bin_pred)

        # c векторами работать легче и нет требований на работу с окрестностями пикселей
        y_true = np.array(etalons_merge).ravel()
        y_pred = np.array(pred_imgs_merge).ravel()

        res[class_name.replace(' ', '_')] = calculateMetrics(y_true, y_pred, using_metrics)
    return res


# Вычисляет все данные метрики для каждого класса со всеми эталонами сразу
def EvaluateMergeImageModelResults(etal_path,
                                   predict_path,
                                   test_img_names,
                                   predict_prefix,
                                   num_classes,
                                   class_names,
                                   using_metrics,
                                   threshold=128):
    # cycle through classes
    model_predicts = []
    for test_img_name in test_img_names:
        predict_data = []
        for i in range(num_classes):
            class_name = class_names[i]

            predict_img_path = os.path.join(predict_path, class_name, predict_prefix + test_img_name)
            pred_img = cv2.imread(predict_img_path, cv2.IMREAD_GRAYSCALE)
            if (pred_img is None):
                print("error predict img")
            predict_data.append(pred_img)
        # переместить каналлы в конец
        predict = np.array(pred_img).transpose(np.roll(np.range(pred_img.ndim), -1))
        model_predicts.append((test_img_name, predict))

    return EvaluateMergeImageModelResultsFromPredict(etal_path, model_predicts, num_classes, class_names, using_metrics, threshold=threshold)


def GetTextMetric(result_mertic_data, using_metric_names, is_print = True, is_all = False):
    model_name = result_mertic_data["Model_name"]
    result_mertic_data = result_mertic_data["Metrics"]

    text_metrics = ""
    text_metrics_all = ""

    classes = {}
    test_img_names = []
    for class_name_and_metrics in result_mertic_data:
        test_img_names.append(class_name_and_metrics.pop("test_img_name"))

        for class_name, metrics in class_name_and_metrics.items():

            if not class_name in classes.keys():
                classes[class_name] = [metrics]
            else:
                classes[class_name].append(metrics)

    title = f"Model: {model_name}\n"
    text_metrics += title
    text_metrics_all += title

    str_metric_name_info = " ".join(using_metric_names)

    metric_name_info = f"\tclass {str_metric_name_info}\n"
    text_metrics += metric_name_info
    text_metrics_all += metric_name_info

    for class_name, metrics_all in classes.items():
        text_metrics_all += f"\t{class_name}"
        for metrics in metrics_all:
            str_metrics = " ".join([f"{metrics[metric_name]:.3f}" for metric_name in using_metric_names])
            text_metrics_all += f"\t\t[{str_metrics}]\n"

    for class_name, metrics_all in classes.items():
        means_vals = []
        for metric_name in using_metric_names:
            mean_metric = None
            num_images = len(metrics_all)
            for metrics in metrics_all:
                if mean_metric is None:
                    mean_metric = metrics[metric_name]
                else:
                    mean_metric += metrics[metric_name]

            mean_metric /= num_images
            means_vals.append(mean_metric)

        str_metrics = " ".join([f"{val:.3f}" for val in means_vals])
        text_metrics += f"\t{class_name} {str_metrics}\n"

    text_metrics = text_metrics.replace(".", ",") + '\n'
    text_metrics_all = text_metrics_all.replace(".", ",") + '\n'
    if is_print:
        if is_all:
            print(text_metrics_all)
        else:
            print(text_metrics)
    return text_metrics, text_metrics_all

def GetFinalTestMetricForExcel(dict_results_mertics_data, using_metric_names, class_names, is_print = True):

    text_metrics = "Model;Metric"
    for class_name in class_names:
        text_metrics += f";{class_name.replace(' ', '_')}"
    text_metrics += '\n'

    for model_test_res in dict_results_mertics_data:
        model_name = model_test_res["Model_name"]
        result_mertic_data = model_test_res["Metrics"]

        #[imgs{classnames[metric[]]}] -> {metric{classnames[imgs]}}
        dict_metrics = {}

        for class_name_and_metrics in result_mertic_data:
            for class_name, metrics in class_name_and_metrics.items():
                for metric_name in using_metric_names:

                    if not metric_name in dict_metrics.keys():
                        dict_metrics[metric_name] = {}

                    if not class_name in dict_metrics[metric_name].keys():
                        dict_metrics[metric_name][class_name] = [metrics[metric_name]]
                    else:
                        dict_metrics[metric_name][class_name].append(metrics[metric_name])

        for metric_name, classes_data in dict_metrics.items():
            text_metrics += f"{model_name};{metric_name}"
            for class_name, metrics in classes_data.items():
                mean_metric = sum(metrics)/len(metrics)

                text_metrics+=f";{mean_metric:.3f}"
            text_metrics+='\n'

    text_metrics = text_metrics.replace(".", ",")
    if is_print:
        print(text_metrics)
    return text_metrics

def CalulateMetricsDir(CNN_name,
                       num_classes,
                       path_train = None,
                       etal_path = "G:/Data/Unet_multiclass/data/original data/testing",
                       predict_prefix = "predict_",
                       class_names = ["mitochondria", "PSD", "vesicles", "axon", "boundaries", "mitochondrial boundaries"],
                       using_metrics = ["Jaccard", "Dice", "RI", "Accuracy", "Precition", "Recall", "Fscore", "CrowdsourcingMetrics"],
                       save_report_path = None,
                       origin_image_path = 'original',
                       path_to_standart_model_result = "data/result/",
                       str_data_time = "2023_05_02",
                       overlap = 128,
                       merge_images = True,
                       is_print_metric = True
                       ):

    if path_train is None:
        path_train = os.path.join(path_to_standart_model_result,
                                   str_data_time,
                                   str(num_classes) + "_class",
                                   CNN_name + "_" + str(overlap))

    if save_report_path is None:
        save_report_path = path_train

    img_suffix = ('.png', '.jpg', '.jpeg')
    etal_image_names = [name for name in os.listdir(os.path.join(etal_path, origin_image_path)) if
                        name.endswith(img_suffix)]
    if len(etal_image_names) == 0:
        print("ERROR !!! NO ETALONS")

    if merge_images:
        res = EvaluateMergeImageModelResults(etal_path,
                                             path_train,
                                             test_img_names=etal_image_names,
                                             predict_prefix=predict_prefix,
                                             num_classes=num_classes,
                                             class_names=class_names,
                                             using_metrics=using_metrics)
        model_results = [res]
    else:
        model_results = []
        for etal_name in etal_image_names:
            model_results_temp = EvaluateSingleImageModelResults(etal_path,
                                                                 path_train,
                                                                 test_img_name=etal_name,
                                                                 predict_prefix=predict_prefix,
                                                                 num_classes=num_classes,
                                                                 class_names=class_names,
                                                                 using_metrics=using_metrics)
            model_results.append(model_results_temp)

    text_result, text_result_all = GetTextMetric(CNN_name, model_results, using_metrics, is_print=is_print_metric)
    test_for_excel = GetFinalTestMetricForExcel({CNN_name: model_results}, using_metrics, class_names, is_print=is_print_metric)

    if save_report_path is not None:
        if not os.path.isdir(save_report_path):
            print(f"create dir:'{save_report_path}'")
            os.makedirs(save_report_path)

        # для длинных путей
        save_report_path = os.path.abspath(save_report_path)
        if os.name == 'nt':  # for Windows
            if save_report_path.startswith(u"\\\\"):
                save_report_path = u"\\\\?\\UNC\\" + save_report_path[2:]
            else:
                save_report_path = u"\\\\?\\" + save_report_path

        with open(os.path.join(save_report_path, f'{CNN_name}{"_merge" if merge_images else ""}_mean.txt'),'w') as file_mean:
            file_mean.write(text_result)
            print(f"{CNN_name}{'_merge' if merge_images else ''}_mean.txt was saved")

        with open(os.path.join(save_report_path, f'{CNN_name}{"_merge" if merge_images else ""}_all.txt'),'w') as file_all:
            file_all.write(text_result_all)
            print(f"{CNN_name}{'_merge' if merge_images else ''}_all.txt was saved")

        with open(os.path.join(save_report_path, f'excel_{CNN_name}{"_merge" if merge_images else ""}.csv'),'w') as file_for_excel:
            file_for_excel.write(test_for_excel)
            print(f"excel_{CNN_name}{'_merge' if merge_images else ''}.csv was saved")

    return model_results, text_result, text_result_all

def CalulateMetricsDirListModels(CNN_names,
                                 list_CNN_num_class,
                                 paths_train = None,
                                 etal_path = "G:/Data/Unet_multiclass/data/original data/testing",
                                 predict_prefix = "predict_",
                                 class_names = ["mitochondria", "PSD", "vesicles", "axon", "boundaries", "mitochondrial boundaries"],
                                 using_metrics = ["Jaccard", "Dice", "RI", "Accuracy", "Precition", "Recall", "Fscore", "CrowdsourcingMetrics"],
                                 save_report_path = "data/report/",
                                 origin_image_path = 'original',
                                 path_to_standart_model_result = "data/result/",
                                 str_data_time = "2023_05_02",
                                 overlap = 128,
                                 merge_images = True,
                                 is_print_metric = True
                                 ):

    num_testing_result = len(list_CNN_num_class)

    all_text_results = ""
    all_text_results_all = ""

    for i in range(num_testing_result):
        save_one_model_info = CNN_names[i]
        num_class = list_CNN_num_class[i]
        model_paths_train = None if paths_train is None else paths_train[i]
        model_save_report_path = None if (save_report_path is None or type(save_report_path) is not list) else save_report_path[i]

        model_results, text_result, text_result_all = CalulateMetricsDir(CNN_name = save_one_model_info,
                                                        num_classes = num_class,
                                                        path_train = model_paths_train,
                                                        etal_path=etal_path,
                                                        predict_prefix=predict_prefix,
                                                        class_names=class_names,
                                                        using_metrics=using_metrics,
                                                        save_report_path=model_save_report_path,
                                                        origin_image_path=origin_image_path,
                                                        path_to_standart_model_result=path_to_standart_model_result,
                                                        str_data_time=str_data_time,
                                                        overlap=overlap,
                                                        merge_images=merge_images,
                                                        is_print_metric = False)

        all_text_results += text_result
        all_text_results_all += text_result_all

    if is_print_metric:
        print(all_text_results)

    if save_report_path is None:
        save_report_path = ''
    else:
        if not os.path.isdir(save_report_path):
            print(f"create dir:'{save_report_path}'")
            os.makedirs(save_report_path)

    with open(os.path.join(save_report_path, f'{str_data_time}_test_models{"_merge" if merge_images else ""}_mean.txt'),'w') as file_mean:
        file_mean.write(all_text_results)

    with open(os.path.join(save_report_path, f'{str_data_time}_test_models{"_merge" if merge_images else ""}_all.txt'),'w') as file_all:
        file_all.write(all_text_results_all)
    print(f"{str_data_time} test was saved to path '{save_report_path}'")

    return all_text_results, all_text_results_all

def CalulateMetricsFromModelPredict(model_predicts,
                                    CNN_name,
                                    num_classes,
                                    etal_path = "G:/Data/Unet_multiclass/data/original data/testing",
                                    class_names = None,
                                    using_metric_names = ["Jaccard", "Dice", "RI", "Accuracy", "Precition", "Recall", "Fscore", "CrowdsourcingMetrics"],
                                    save_report_path = None,
                                    merge_images = True,
                                    is_print_metric = True
                                    ):



    using_metrics = [(METRIC_FUN[name], METRIC_NAMES[name]) for name in using_metric_names]

    model_results = {"Model_name": CNN_name}
    if merge_images:
        res = EvaluateMergeImageModelResultsFromPredict(etal_path,
                                                        model_predicts,
                                                        num_classes=num_classes,
                                                        class_names=class_names,
                                                        using_metrics=using_metrics)

        model_results["Metrics"] = [res]
    else:
        model_results["Metrics"] = []
        for one_model_predict in model_predicts:
            model_results_temp = EvaluateSingleImageModelResultsFromPredict(etal_path,
                                                                            one_model_predict,
                                                                            num_classes=num_classes,
                                                                            class_names=class_names,
                                                                            using_metrics=using_metrics)
            model_results["Metrics"].append(model_results_temp)

    text_result, text_result_all = GetTextMetric(model_results, using_metric_names, is_print=is_print_metric)
    test_for_excel = GetFinalTestMetricForExcel([model_results], using_metric_names, class_names, is_print=is_print_metric)

    if save_report_path is not None:
        if not os.path.isdir(save_report_path):
            print(f"create dir:'{save_report_path}'")
            os.makedirs(save_report_path)

        # для длинных путей
        if os.name == 'nt':  # for Windows
            save_report_path = os.path.abspath(save_report_path)
            if save_report_path.startswith(u"\\\\"):
                save_report_path = u"\\\\?\\UNC\\" + save_report_path[2:]
            else:
                save_report_path = u"\\\\?\\" + save_report_path

        with open(os.path.join(save_report_path, f'{CNN_name}{"_merge" if merge_images else ""}_mean.txt'),'w') as file_mean:
            file_mean.write(text_result)
            print(f"{CNN_name}{'_merge' if merge_images else ''}_mean.txt was saved")

        with open(os.path.join(save_report_path, f'{CNN_name}{"_merge" if merge_images else ""}_all.txt'),'w') as file_all:
            file_all.write(text_result_all)
            print(f"{CNN_name}{'_merge' if merge_images else ''}_all.txt was saved")

        with open(os.path.join(save_report_path, f'excel_{CNN_name}{"_merge" if merge_images else ""}.csv'),'w') as file_for_excel:
            file_for_excel.write(test_for_excel)
            print(f"excel_{CNN_name}{'_merge' if merge_images else ''}.csv was saved")

    return model_results, text_result, text_result_all

def CalulateMetricsFromModelPredictAndEtalData(model_predicts,
                                               data_names,
                                               CNN_name,
                                               num_classes,
                                               etal_data,
                                               class_names = None,
                                               using_metric_names = ["Jaccard", "Dice", "RI", "Accuracy", "Precition", "Recall", "Fscore", "CrowdsourcingMetrics"],
                                               save_report_path = None,
                                               merge_images = True,
                                               is_print_metric = True
                                               ):



    using_metrics = [(METRIC_FUN[name], METRIC_NAMES[name]) for name in using_metric_names]

    model_results = {"Model_name": CNN_name}
    if merge_images:
        res = EvaluateMergeImageModelResultsFromPredictAndEtal(etal_data,
                                                               model_predicts,
                                                               num_classes=num_classes,
                                                               class_names=class_names,
                                                               using_metrics=using_metrics)

        model_results["Metrics"] = [res]
    else:
        model_results["Metrics"] = []
        for i, one_model_predict in enumerate(model_predicts):
            model_results_temp = EvaluateSingleImageModelResultsFromPredictAndEtal(etal_data,
                                                                                   data_names[i],
                                                                                   one_model_predict,
                                                                                   num_classes=num_classes,
                                                                                   class_names=class_names,
                                                                                   using_metrics=using_metrics)
            model_results["Metrics"].append(model_results_temp)

    text_result, text_result_all = GetTextMetric(model_results, using_metric_names, is_print=is_print_metric)
    test_for_excel = GetFinalTestMetricForExcel([model_results], using_metric_names, class_names, is_print=is_print_metric)

    if save_report_path is not None:
        if not os.path.isdir(save_report_path):
            print(f"create dir:'{save_report_path}'")
            os.makedirs(save_report_path)

        # для длинных путей
        if os.name == 'nt':  # for Windows
            save_report_path = os.path.abspath(save_report_path)
            if save_report_path.startswith(u"\\\\"):
                save_report_path = u"\\\\?\\UNC\\" + save_report_path[2:]
            else:
                save_report_path = u"\\\\?\\" + save_report_path

        with open(os.path.join(save_report_path, f'{CNN_name}{"_merge" if merge_images else ""}_mean.txt'),'w') as file_mean:
            file_mean.write(text_result)
            print(f"{CNN_name}{'_merge' if merge_images else ''}_mean.txt was saved")

        with open(os.path.join(save_report_path, f'{CNN_name}{"_merge" if merge_images else ""}_all.txt'),'w') as file_all:
            file_all.write(text_result_all)
            print(f"{CNN_name}{'_merge' if merge_images else ''}_all.txt was saved")

        with open(os.path.join(save_report_path, f'excel_{CNN_name}{"_merge" if merge_images else ""}.csv'),'w') as file_for_excel:
            file_for_excel.write(test_for_excel)
            print(f"excel_{CNN_name}{'_merge' if merge_images else ''}.csv was saved")

    return model_results, text_result, text_result_all

if __name__ == "__main__":
    str_data = "2023_05_01"

    classnames = ["mitochondria", "PSD", "vesicles", "axon", "boundaries", "mitochondrial boundaries"]

    CNN_name = [
        "model_by_config_sint_Lars76_unet",
        "model_by_config_Lars76_unet"
    ]

    standart_path_to_model_result = "../data/result/"

    list_CNN_num_class = [
        6,
        6
    ]

    overlap = 128

    CalulateMetricsDirListModels(CNN_name,
                                 list_CNN_num_class,
                                 str_data_time=str_data,
                                 overlap=overlap,
                                 class_names = classnames,
                                 using_metrics = [Jaccard, Dice],
                                 path_to_standart_model_result=standart_path_to_model_result,
                                 merge_images = False
                                 )

    CalulateMetricsDirListModels(CNN_name,
                                 list_CNN_num_class,
                                 str_data_time=str_data,
                                 overlap=overlap,
                                 class_names = classnames,
                                 using_metrics=[Jaccard, Dice],
                                 path_to_standart_model_result=standart_path_to_model_result,
                                 merge_images=True
                                 )