import numpy as np
import cv2
import os
import json
import math

import sklearn.metrics

############################################### доделать сохранение метрик в файл

def to_0_255_format_img(in_img):
    max_val = in_img[:, :].max()
    if max_val <= 1:
        out_img = np.round(in_img * 255)
        return out_img.astype(np.uint8)
    else:
        return in_img

def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def Jaccard(y_true, y_pred):
    error = 0.0000001
    y_true_bool = np.asarray(y_true, bool)  # Not necessary, if you keep your data
    y_pred_bool = np.asarray(y_pred, bool)  # in a boolean array already!

    intersection = np.double(np.bitwise_and(y_true_bool, y_pred_bool).sum())
    union = np.double(np.bitwise_or(y_true_bool, y_pred_bool).sum())
    return (intersection + error) / (union + error)

def Dice(y_true, y_pred):
    error = 0.0000001
    y_true_bool = np.asarray(y_true, bool)  # Not necessary, if you keep your data
    y_pred_bool = np.asarray(y_pred, bool)  # in a boolean array already!
    intersection = np.double(np.bitwise_and(y_true_bool, y_pred_bool).sum())
    union_and_intersection = y_true_bool.sum() + y_pred_bool.sum()
    return (2. * intersection + error) / (union_and_intersection + error)

def RI(y_true, y_pred):
    try:
        y_true = np.asarray(y_true, bool).astype(np.int32)
        y_pred = np.asarray(y_pred, bool).astype(np.int32)
        TN, FP, FN, TP = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
        n = len(y_true)
        a = 0.5 * (TP * (TP - 1) + FP * (FP - 1) + TN * (TN - 1) + FN * (FN - 1))
        b = 0.5 * ((TP + FN) ** 2 + (TN + FP) ** 2 - (TP ** 2 + TN ** 2 + FP ** 2 + FN ** 2))
        c = 0.5 * ((TP + FP) ** 2 + (TN + FN) ** 2 - (TP ** 2 + TN ** 2 + FP ** 2 + FN ** 2))
        d = n * (n - 1) / 2 - (a + b + c)

        RI = (a + b) / (a + b + c + d)
    except:
        print("RI EXEPTION")
        RI = 0

    return RI

def Accuracy(y_true, y_pred):
    try:
        y_true = np.asarray(y_true, bool).astype(np.int32)
        y_pred = np.asarray(y_pred, bool).astype(np.int32)
        TN, FP, FN, TP = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
        accuracy = float(TN + TP) / (TN + TP + FN + FP)
    except:
        print("Accuracy EXEPTION")
        accuracy = 0
    return accuracy

def Precition(y_true, y_pred):
    try:
        y_true = np.asarray(y_true, bool).astype(np.int32)
        y_pred = np.asarray(y_pred, bool).astype(np.int32)
        TN, FP, FN, TP = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
        precition = float(TP) / (TP + FP)
    except:
        print("Precition EXEPTION")
        precition = 0
    return precition

def Recall(y_true, y_pred):
    try:
        y_true = np.asarray(y_true, bool).astype(np.int32)
        y_pred = np.asarray(y_pred, bool).astype(np.int32)
        TN, FP, FN, TP = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
        recall = float(TP) / (TP + FN)
    except:
        print("Recall EXEPTION")
        recall = 0
    return recall

def Fscore(y_true, y_pred):
    try:
        y_true = np.asarray(y_true, bool).astype(np.int32)
        y_pred = np.asarray(y_pred, bool).astype(np.int32)
        precition = Precition(y_true, y_pred)
        recall = Recall(y_true, y_pred)

        fscore = (2 * precition * recall) / (precition + recall)
    except:
        print("Fscore EXEPTION")
        fscore = 0
    return fscore

def CrowdsourcingMetrics(y_true, y_pred):
    y_true = np.asarray(y_true, bool).astype(np.int32).ravel()
    y_pred = np.asarray(y_pred, bool).astype(np.int32).ravel()
    n = len(y_true)
    num_class = 1
    pij_matrix = np.zeros((num_class + 1, num_class + 1), np.float64)

    for i in range(len(y_true)):
        pij_matrix[y_pred[i], y_true[i]] += 1
    pij_matrix = pij_matrix / n  # pij_matrix.sum()

    s_i_arr = np.zeros(pij_matrix.shape[0], np.float64)
    for i in range(0, pij_matrix.shape[0]):
        for j in range(pij_matrix.shape[1]):
            s_i_arr[i] += pij_matrix[i][j]

    t_j_arr = np.zeros(pij_matrix.shape[1], np.float64)
    for j in range(0, pij_matrix.shape[1]):
        for i in range(0, pij_matrix.shape[0]):
            t_j_arr[j] += pij_matrix[i][j]

    sqr_t_sum = (t_j_arr ** 2).sum()
    sqr_s_sum = (s_i_arr ** 2).sum()
    sqr_pij_sum = (pij_matrix ** 2).sum()

    Vrand_split = sqr_pij_sum / sqr_t_sum
    Vrand_merge = sqr_pij_sum / sqr_s_sum

    Rand_Fscore = 2.0 * sqr_pij_sum / (sqr_t_sum + sqr_s_sum)

    p_logp = 0
    for i in range(0, pij_matrix.shape[0]):
        for j in range(0, pij_matrix.shape[1]):
            if pij_matrix[i, j] != 0:
                p_logp += pij_matrix[i, j] * math.log(pij_matrix[i, j])
    s_logs = 0
    for s_i in s_i_arr[:]:
        if s_i != 0:
            s_logs -= s_i * math.log(s_i)
    t_logt = 0
    for t_j in t_j_arr[:]:
        if t_j != 0:
            t_logt -= t_j * math.log(t_j)

    I = p_logp + s_logs + t_logt

    Vinfo_split = I / (s_logs)
    Vinfo_merge = I / (t_logt)

    InformationTheoreticFscore = 2.0 * I / (s_logs + t_logt)

    return [Vrand_split, Vrand_merge, Rand_Fscore, Vinfo_split, Vinfo_merge, InformationTheoreticFscore]

def calculateMetric(y_true, y_pred, metrics = []):
    result = []
    for metric in metrics:
        if metric.__name__ == "CrowdsourcingMetrics":
            temp_result = metric(y_true, y_pred)
            result += temp_result
        else:
            result.append(metric(y_true, y_pred))
    return result

# Вычисляет все данные метрики для каждого класса одного реального изображения
def EvaluateSingleImageModelResults(etal_path, predict_path, test_img_name, predict_prefix, num_classes, class_names, using_metrics):
    result = {}
    # cycle through classes
    for i in range(num_classes):
        class_name = class_names[i]

        etal_img_path = os.path.join(etal_path, class_name, test_img_name)
        etal = cv2.imread(etal_img_path, cv2.IMREAD_GRAYSCALE)
        etal = to_0_255_format_img(etal)
        if (etal is None):
            print("error etal")

        predict_img_path = os.path.join(predict_path, class_name, predict_prefix+test_img_name)
        pred_img = cv2.imread(predict_img_path, cv2.IMREAD_GRAYSCALE)
        pred_img = to_0_255_format_img(pred_img)
        if (pred_img is None):
            print("error predict img")

        # бинаризация с порогом (на всякий случай)
        threshold = 128
        ret, bin_true = cv2.threshold(etal, threshold, 255, 0)
        ret, bin_img_true = cv2.threshold(pred_img, threshold, 255, 0)

        # c векторами работать легче и нет требований на работу с окрестностями пикселей
        y_true = bin_true.ravel()
        y_pred = bin_img_true.ravel()

        result[class_name.replace(' ', '_')] = calculateMetric(y_true, y_pred, using_metrics)
    return result

# Вычисляет все данные метрики для каждого класса со всеми эталонами сразу
def EvaluateMergeImageModelResults(etal_path, predict_path, test_img_names, predict_prefix, num_classes, class_names, using_metrics):
    result = {}
    # cycle through classes
    for i in range(num_classes):
        class_name = class_names[i]

        etalons_merge = []
        pred_imgs_merge = []

        for test_img_name in test_img_names:
            etal_img_path = os.path.join(etal_path, class_name, test_img_name)
            etal = cv2.imread(etal_img_path, cv2.IMREAD_GRAYSCALE)
            etal = to_0_255_format_img(etal)
            if (etal is None):
                print("error etal")

            predict_img_path = os.path.join(predict_path, class_name, predict_prefix+test_img_name)
            pred_img = cv2.imread(predict_img_path, cv2.IMREAD_GRAYSCALE)
            pred_img = to_0_255_format_img(pred_img)
            if (pred_img is None):
                print("error predict img")

            # бинаризация с порогом (на всякий случай)
            threshold = 128
            ret, bin_true = cv2.threshold(etal, threshold, 255, 0)
            ret, bin_img_true = cv2.threshold(pred_img, threshold, 255, 0)

            etalons_merge.append(bin_true)
            pred_imgs_merge.append(bin_img_true)

        # c векторами работать легче и нет требований на работу с окрестностями пикселей
        y_true = np.array(etalons_merge).ravel()
        y_pred = np.array(pred_imgs_merge).ravel()

        result[class_name.replace(' ', '_')] = calculateMetric(y_true, y_pred, using_metrics)
    return result


def GetTestMetric(model_name, result_mertic_data, using_metrics, is_print = True, is_all = False):

    text_metrics = ""
    text_metrics_all = ""

    classes = {}
    for class_name_and_metrics in result_mertic_data:
        for class_name, metrics in class_name_and_metrics.items():
            if not class_name in classes.keys():
                classes[class_name] = [metrics]
            else:
                classes[class_name].append(metrics)

    title = f"Model: {model_name}\n"
    text_metrics += title
    text_metrics_all += title

    str_metric_name_info = " ".join([metric_name.__name__ for metric_name in using_metrics])

    metric_name_info = f"\tclass {str_metric_name_info}\n"
    text_metrics += metric_name_info
    text_metrics_all += metric_name_info


    for class_name, metrics_all in classes.items():
        text_metrics_all += f"\t{class_name}"
        for metrics in metrics_all:
            str_metrics = " ".join([f"{val:.3f}" for val in metrics])
            text_metrics_all += f"\t\t[{str_metrics}]\n"

    for class_name, metrics_all in classes.items():
        mean_metric = None
        num_images = len(metrics_all)
        for metrics in metrics_all:
            if mean_metric is None:
                mean_metric = metrics
            else:
                for i, metric in enumerate(metrics):
                    mean_metric[i] += metric

        for i in range(len(mean_metric)):
            mean_metric[i] /= num_images

        str_metrics = " ".join([f"{val:.3f}" for val in mean_metric])
        text_metrics += f"\t{class_name} {str_metrics}\n"

    text_metrics = text_metrics.replace(".", ",") + '\n'
    text_metrics_all = text_metrics_all.replace(".", ",") + '\n'
    if is_print:
        if is_all:
            print(text_metrics_all)
        else:
            print(text_metrics)
    return text_metrics, text_metrics_all

def CalulateMetricsDir(CNN_name,
                       num_classes,
                       path_train = None,
                       etal_path = "G:/Data/Unet_multiclass/data/original data/testing",
                       predict_prefix = "predict_",
                       class_names = ["mitochondria", "PSD", "vesicles", "axon", "boundaries", "mitochondrial boundaries"],
                       using_metrics = [Jaccard, Dice, RI, Accuracy, Precition, Recall, Fscore, CrowdsourcingMetrics],
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

    text_result, text_result_all = GetTestMetric(CNN_name, model_results, using_metrics, is_print=is_print_metric)
    return model_results, text_result, text_result_all

def CalulateMetricsDirs(CNN_names,
                        list_CNN_num_class,
                        paths_train = None,
                        etal_path = "G:/Data/Unet_multiclass/data/original data/testing",
                        predict_prefix = "predict_",
                        class_names = ["mitochondria", "PSD", "vesicles", "axon", "boundaries", "mitochondrial boundaries"],
                        using_metrics = [Jaccard, Dice, RI, Accuracy, Precition, Recall, Fscore, CrowdsourcingMetrics],
                        save_report_path = None,
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
        model_save_report_path = None if save_report_path is None else save_report_path[i]

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
    # save to path_to_standart_model_result +/+ str_data_time

if __name__ == "__main__":
    str_data = "2023_05_02"

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

    CalulateMetricsDirs(CNN_name,
                        list_CNN_num_class,
                        str_data_time=str_data,
                        overlap=overlap,
                        class_names = classnames,
                        using_metrics = [Jaccard, Dice],
                        path_to_standart_model_result=standart_path_to_model_result,
                        merge_images = False
                        )

    CalulateMetricsDirs(CNN_name,
                        list_CNN_num_class,
                        str_data_time=str_data,
                        overlap=overlap,
                        class_names = classnames,
                        using_metrics=[Jaccard, Dice],
                        path_to_standart_model_result=standart_path_to_model_result,
                        merge_images=True
                        )