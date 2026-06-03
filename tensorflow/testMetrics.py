import numpy as np
import cv2
import copy
import os
import json
import math

import skimage.io as io

import sklearn.metrics


def to_0_255_format_img(in_img):
    max_val = in_img[:,:].max()
    if max_val <= 1:
       out_img = np.round(in_img * 255)
       return out_img.astype(np.uint8)
    else:
        return in_img

def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def jaccard(x,y):
  error = 0.0000001
  x = np.asarray(x, bool) # Not necessary, if you keep your data
  y = np.asarray(y, bool) # in a boolean array already!
  return np.double(np.bitwise_and(x, y).sum()+error) / np.double(np.bitwise_or(x, y).sum()+error)

def dice(y_true, y_pred):
    error = 0.0000001
    y_true = np.asarray(y_true, bool)  # Not necessary, if you keep your data
    y_pred = np.asarray(y_pred, bool)  # in a boolean array already!
    intersection = np.double(np.bitwise_and(y_true, y_pred).sum())
    #print(2 * intersection, len(y_true) + len(y_pred), 1024*768)
    #print(intersection)
    return (2. * intersection + error) / ((y_true.sum()) + (y_pred.sum())+ error)
    
def RI(y_true, y_pred):
    try:
        y_true = np.asarray(y_true, bool)
        y_pred = np.asarray(y_pred, bool)
        
        y_true = np.asarray(y_true, np.int32)
        y_pred = np.asarray(y_pred, np.int32)
        
        #print(y_true.sum())
        #print(y_pred.sum())
            
        TN, FP, FN, TP = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
        
        #print (TN, FP, FN, TP)
        
        n = len(y_true)
        
        a = 0.5 *(TP*(TP-1)+FP*(FP-1)+TN*(TN-1)+FN*(FN-1))
        
        b = 0.5 *((TP+FN)**2 + (TN+FP)**2 - (TP**2+ TN**2+ FP**2+ FN**2))
        
        #print(TP, TP**2, TN, TN**2, FP, FP**2, FN, FN**2)
        
        c = 0.5 *((TP+FP)**2 + (TN+FN)**2 - (TP**2+ TN**2+ FP**2+ FN**2))
        
        d = n*(n-1)/2 - (a+b+c)
            
        RI = (a+b)/(a+b+c+d)
    except:
        print("RI EXEPTION")
        RI = 0
        
    return RI

def Accuracy(y_true, y_pred):

    try:
        y_true = np.asarray(y_true, bool)
        y_pred = np.asarray(y_pred, bool)
        
        y_true = np.asarray(y_true, np.int32)
        y_pred = np.asarray(y_pred, np.int32)
        
        #print(y_true.sum())
        #print(y_pred.sum())
            
        TN, FP, FN, TP = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
            
        accuracy = float(TN + TP)/(TN + TP+ FN + FP)
            
    except:
        print("Accuracy EXEPTION")
        accuracy = 0
         
      
    return accuracy
    
def Precition(y_true, y_pred):
    
    try:
        y_true = np.asarray(y_true, bool)
        y_pred = np.asarray(y_pred, bool)
        
        y_true = np.asarray(y_true, np.int32)
        y_pred = np.asarray(y_pred, np.int32)
                
        TN, FP, FN, TP = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
        
        precition = float(TP)/(TP+FP)
        
    except:
        print("Precition EXEPTION")
        precition = 0
        
    return precition
    
def Recall(y_true, y_pred):

    try:
        y_true = np.asarray(y_true, bool)
        y_pred = np.asarray(y_pred, bool)
        
        y_true = np.asarray(y_true, np.int32)
        y_pred = np.asarray(y_pred, np.int32)
                
        TN, FP, FN, TP = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
        
        recall = float(TP)/(TP+FN)
    except:
        print("Recall EXEPTION")
        recall = 0
                
    return recall
    
def Fscore(y_true, y_pred):

    try:

        y_true = np.asarray(y_true, bool)
        y_pred = np.asarray(y_pred, bool)
        
        y_true = np.asarray(y_true, np.int32)
        y_pred = np.asarray(y_pred, np.int32)
        
        #print(y_true.sum())
        #print(y_pred.sum())
        precition = Precition(y_true, y_pred)    
        recall = Recall(y_true, y_pred) 

        fscore = (2*precition*recall)/(precition+recall)
    
    except:
        print("Fscore EXEPTION")
        fscore = 0
                
    return fscore


def CrowdsourcingMetrics(y_true, y_pred):
    y_true = np.asarray(y_true, bool)
    y_pred = np.asarray(y_pred, bool)

    y_true = np.asarray(y_true, np.int16)
    y_pred = np.asarray(y_pred, np.int16)

    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    n = len(y_true)
    num_class = 1

    pij_matrix = np.zeros((num_class + 1, num_class + 1), np.float64)

    for i in range(len(y_true)):
        pij_matrix[y_pred[i], y_true[i]] += 1

    # print (pij_matrix)
    # print (n, pij_matrix.sum(), pij_matrix.sum()/n)
    pij_matrix = pij_matrix / n  # pij_matrix.sum()
    # print (pij_matrix)
    # print (n, pij_matrix.sum())

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

    # print(s_i_arr, t_j_arr)
    # print(sqr_t_sum, sqr_s_sum)
    # print(pij_matrix**2)
    # print(sqr_pij_sum, "***")

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

def TestsMetric():
    mask_name_label_list = ["mitochondria", "PSD", "vesicles", "axon", "boundaries", "mitochondrial boundaries"]

    name_img = "testing0000.png"

    list_CNN_num_class = [6]

    result_CNN_dir = ["data/result/14_01_2022_klaster_new_unet_train_with_sintetik_and_agcwd_6_classes_64",
                      "data/result/my_unet_multidata_pe70_bs10_6class_no_test_v9_last_image_one",
                      "data/result/my_unet_multidata_pe69_bs9_6class_no_test_v8_100ep_image_one",
                      "data/result/CNN_5_class_with_test_128"
                      ]

    result_CNN_json_name = ["CNN_6_class_64",
                            "CNN_6_class_image_one",
                            "CNN_6_class_image_one",
                            "CNN_5_class_with_test_128"
                            ]

    for i in range(len(list_CNN_num_class)):
        num_class = list_CNN_num_class[i]
        print(result_CNN_json_name[i])

        print("class\metrics", "jaccard", "dice", "RI", "Accuracy", "AdaptedRandError", "Fscore")
        json_list = [["class\metrics", "jaccard", "dice", "RI", "Accuracy", "AdaptedRandError", "Fscore", "Vrand_split",
                      "Vrand_merge", "Rand_Fscore", "Vinfo_split", "Vinfo_merge", "InformationTheoreticFscore"]]
        for index_label_name in range(num_class):
            original_name = os.path.join("data/original data/testing/", mask_name_label_list[index_label_name],
                                         name_img)
            etal = io.imread(original_name, as_gray=True)
            etal = to_0_255_format_img(etal)
            if (etal.size == 0):
                print("error etal")

            test_img_name = "predict_" + name_img

            test_img_dir = os.path.join(result_CNN_dir[i], mask_name_label_list[index_label_name], test_img_name)
            img = io.imread(test_img_dir, as_gray=True)

            img = to_0_255_format_img(img)

            if (img.size == 0):
                print("error img")

            ret, bin_true = cv2.threshold(etal, 128, 255, 0)
            ret, bin_img_true = cv2.threshold(img, 128, 255, 0)

            # print(img)
            # print(etal)
            # viewImage(bin_true,"etal")
            # viewImage(bin_img_true,"img")

            y_true = bin_true.ravel()
            y_pred = bin_img_true.ravel()

            # blac
            ret, bin_pred1 = cv2.threshold(etal, 0, 0, 0)
            y_pred1 = bin_pred1.ravel()
            # Brez = jaccard_similarity_score(y_true, y_pred1)

            # white
            ret, bin_pred2 = cv2.threshold(etal, 255, 255, 1)
            y_pred2 = bin_pred2.ravel()
            # Wrez = jaccard_similarity_score(y_true, y_pred2)

            test = jaccard(y_true, y_true)
            Brez2 = jaccard(y_true, y_pred1)
            Wrez2 = jaccard(y_true, y_pred2)

            # viewImage(y_pred1,"bitB")
            # viewImage(y_pred2,"bitW")

            #
            # print(Brez,Wrez)
            # print(Brez2,Wrez2, test)
            # cv2.waitKey(0)

            rez = jaccard(y_true, y_pred)
            rez2 = dice(y_true, y_pred)
            res3 = RI(y_true, y_pred)
            res4 = Accuracy(y_true, y_pred)
            res5 = Fscore(y_true, y_pred)  # Adapted Rand Error

            Vrand_split, Vrand_merge, Rand_Fscore, Vinfo_split, Vinfo_merge, InformationTheoreticFscore = CrowdsourcingMetrics(
                y_true, y_pred)
            print(mask_name_label_list[index_label_name], rez, rez2, res3, res4, 1 - res5, res5)
            print("Vrand_split", "Vrand_merge", "Rand_Fscore", "Vinfo_split", "Vinfo_merge", "InformationTheoreticFscore")
            print(Vrand_split, Vrand_merge, Rand_Fscore, Vinfo_split, Vinfo_merge, InformationTheoreticFscore)

            json_list.append(
                [mask_name_label_list[index_label_name], rez, rez2, res3, res4, 1 - res5, res5, Vrand_split,
                 Vrand_merge, Rand_Fscore, Vinfo_split, Vinfo_merge, InformationTheoreticFscore])

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        with open(result_CNN_dir[i] + "/result_" + result_CNN_json_name[i] + ".json", 'w') as file:
            json.dump(json_list, file)


def deleteZero_and_predict_mask(name):
    rename = name.replace('predict_','')
    while rename[0] == '0' and len(rename) > 5:
        rename = rename[1:]
    return rename


def TestsMetricDir(data = None, CNN_name = None, list_CNN_num_class = None, overlap = 64):
    mask_name_label_list = ["mitochondria", "PSD", "vesicles", "axon", "boundaries", "mitochondrial boundaries"]

    if data is None:
        data = "2022_11_16"

    if CNN_name is None:
        CNN_name = [
                    "real_data_256_full_unet_6_num_class_27_slices",
                    "real_data_256_tiny_unet_5_num_class_27_slices",
                    "real_data_256_tiny_unet_5_num_class_27_slices_AdamW",
                    "real_data_256_tiny_unet_5_num_class_27_slices_AdamW_v2",
                    "real_data_256_tiny_unet_5_num_class_27_slices_AdamW_v3",
                    "real_data_256_tiny_unet_5_num_class_27_slices_v3",
                    "real_data_256_tiny_unet_6_num_class_27_slices",
                    "real_data_256_tiny_unet_6_num_class_27_slices_AdamW",
                    "real_data_256_tiny_unet_6_num_class_27_slices_AdamW_v2",
                    "real_data_256_tiny_unet_6_num_class_27_slices_AdamW_v3",
                    "real_data_256_tiny_unet_6_num_class_27_slices_no_noise_no_zoom",
                    "real_data_256_tiny_unet_6_num_class_27_slices_v3",
                    "real_data_256_unet_6_num_class_27_slices_no_noise_no_zoom_v2",
                    "real_data_256_unet_6_num_class_27_slices_no_noise_no_zoom_v3_new_gen",
                    "real_data_256_unet_6_num_class_27_slices_no_noise_no_zoom_v3_new_gen_adamW",
                    "real_data_256_unet_6_num_class_27_slices_no_noise_no_zoom_v3_new_gen_adamW_lr0001",
                    "real_data_256_unet_6_num_class_27_slices_no_noise_no_zoom_v3_new_gen_adamW_lr001",
                    "real_data_256_unet_6_num_class_27_slices_no_noise_no_zoom_v3_new_gen_adamW_lr001_v2",
                    "real_data_256_unet_6_num_class_27_slices_no_noise_no_zoom_v3_new_gen_novorad"                      
                   ]

    if list_CNN_num_class is None:
        list_CNN_num_class = [
                              6,
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
                              6,
                              6,
                              6,
                              6,
                              6,
                              6,
                              6,
                              6,
                              ]

    result_CNN_dir = []
    
    for i in range(len(list_CNN_num_class)):
        save_name = "data/result/" + data + "/" + str(list_CNN_num_class[i]) + "_class/" + CNN_name[i]
        result_CNN_dir.append(save_name)

    result_CNN_json_name = []
    
    for i in range(len(list_CNN_num_class)):
        CNN_json_name = "CNN_" + str(list_CNN_num_class[i])
        result_CNN_json_name.append(CNN_json_name)


    save_name_and_dice_result = []

    for i in range(len(list_CNN_num_class)):
    
        save_one_model_info = [CNN_name[i]]

        num_class = list_CNN_num_class[i]
        print(result_CNN_json_name[i])

        print("class\metrics", "jaccard", "dice", "RI", "Accuracy", "AdaptedRandError", "Fscore")
        str_tabel = ""
        for one_str in ["class\metrics", "jaccard", "dice", "RI", "Accuracy", "AdaptedRandError", "Fscore", "Vrand_split","Vrand_merge", "Rand_Fscore", "Vinfo_split", "Vinfo_merge", "InformationTheoreticFscore"]:
            str_tabel += one_str + " "
        str_tabel += '\n'
        str_tabel_all = str_tabel

        index_name = 0
        for index_label_name in range(num_class):
            list_name = os.listdir(os.path.join(result_CNN_dir[i] + "_" + str(overlap), mask_name_label_list[index_label_name]))
            json_temp = []
            for num,testName in enumerate(list_name):
                print(num + 1, "image is ", len(list_name))

                dir_etal = os.path.join("data", "original data/testing", mask_name_label_list[index_label_name], deleteZero_and_predict_mask(testName))

                etal = io.imread(dir_etal, as_gray=True)
                etal = to_0_255_format_img(etal)
                if (etal.size == 0):
                    print("error etal")

                test_img_name = "predict_" + testName

                test_img_dir = os.path.join(os.path.join(result_CNN_dir[i]+ "_" + str(overlap), mask_name_label_list[index_label_name]), testName)

                img = io.imread(test_img_dir, as_gray=True)

                img = to_0_255_format_img(img)

                if (img.size == 0):
                    print("error img")

                ret, bin_true = cv2.threshold(etal, 128, 255, 0)
                ret, bin_img_true = cv2.threshold(img, 128, 255, 0)

                # print(img)
                # print(etal)
                # viewImage(bin_true,"etal")
                # viewImage(bin_img_true,"img")

                y_true = bin_true.ravel()
                y_pred = bin_img_true.ravel()

                # blac
                #ret, bin_pred1 = cv2.threshold(etal, 0, 0, 0)
                #y_pred1 = bin_pred1.ravel()
                # Brez = jaccard_similarity_score(y_true, y_pred1)

                # white
                #ret, bin_pred2 = cv2.threshold(etal, 255, 255, 1)
                #y_pred2 = bin_pred2.ravel()
                # Wrez = jaccard_similarity_score(y_true, y_pred2)

                #test = jaccard(y_true, y_true)
                #Brez2 = jaccard(y_true, y_pred1)
                #Wrez2 = jaccard(y_true, y_pred2)

                # viewImage(y_pred1,"bitB")
                # viewImage(y_pred2,"bitW")

                #
                # print(Brez,Wrez)
                # print(Brez2,Wrez2, test)
                # cv2.waitKey(0)

                rez = jaccard(y_true, y_pred)
                rez2 = dice(y_true, y_pred)
                res3 = RI(y_true, y_pred)
                res4 = Accuracy(y_true, y_pred)
                res5 = Fscore(y_true, y_pred)  # Adapted Rand Error

                Vrand_split, Vrand_merge, Rand_Fscore, Vinfo_split, Vinfo_merge, InformationTheoreticFscore = CrowdsourcingMetrics(
                    y_true, y_pred)

                # print("Vrand_split", "Vrand_merge", "Rand_Fscore", "Vinfo_split", "Vinfo_merge", "InformationTheoreticFscore")
                # print(Vrand_split, Vrand_merge, Rand_Fscore, Vinfo_split, Vinfo_merge, InformationTheoreticFscore)

                if len (json_temp) == 0:
                    json_temp.append([mask_name_label_list[index_label_name].replace(' ', '_'), rez, rez2, res3, res4, 1 - res5, res5, Vrand_split, Vrand_merge, Rand_Fscore, Vinfo_split, Vinfo_merge, InformationTheoreticFscore])

                else:
                    t_list = [mask_name_label_list[index_label_name].replace(' ', '_'), rez, rez2, res3, res4, 1 - res5, res5, Vrand_split, Vrand_merge, Rand_Fscore, Vinfo_split, Vinfo_merge, InformationTheoreticFscore]
                    for j in range(1, len(json_temp[0])):
                        json_temp[0][j] += t_list[j]

                for one_str in [index_name, mask_name_label_list[index_label_name].replace(' ', '_'), rez, rez2, res3, res4, 1 - res5, res5, Vrand_split, Vrand_merge, Rand_Fscore, Vinfo_split, Vinfo_merge, InformationTheoreticFscore]:
                    str_tabel_all += str(one_str) + " "
                str_tabel_all += '\n'

                index_name += 1

            for j in range(1,len(json_temp[0])):
                json_temp[0][j] /= len(list_name)

            for len_str in json_temp:
                save_one_model_info.append(len_str)
                for one_str in len_str:
                    str_tabel += str(one_str) + " "
                str_tabel += '\n'
                

            with open(result_CNN_dir[i]+ "_" + str(overlap) + "/result_" + result_CNN_json_name[i] + ".txt", 'w') as file_mean:
                file_mean.write(str_tabel)
            with open(result_CNN_dir[i]+ "_" + str(overlap) + "/result_" + result_CNN_json_name[i] + "ALL.txt", 'w') as file_all:
                file_all.write(str_tabel_all)
                
        save_name_and_dice_result.append(save_one_model_info)

    
    
    for save_one_model_info in save_name_and_dice_result:
        model_name = save_one_model_info[0]
        len_classes = len(save_one_model_info) - 1

        len_name = len(model_name)
            
        num_tab = (50 - len_name) // 4
        
        str_tab = ''
        for t in range(num_tab):
            str_tab += '\t'
       
        str_dice = ''
        
        for c in range(len_classes):
            str_dice +=  " " + str(save_one_model_info[c+1][2])

        print (model_name + str_tab + str_dice)
        
    all_names = 'class/name'
    for i in range(len(save_name_and_dice_result)):
        all_names += save_name_and_dice_result[i][0] + " "
    print(all_names)
    
    for k in range(6):
        class_info = "class_" + str(k) + ": "
        for i in range(len(save_name_and_dice_result)):
            if len(save_name_and_dice_result[i]) - 1 > k:
                class_info += str(save_name_and_dice_result[i][k+1][2]) + " "
            else:
                class_info += " "
        
        print(class_info) 


def TestsMetricDirDice(data = None, CNN_name = None, list_CNN_num_class = None, overlap = 64):
    mask_name_label_list = ["mitochondria", "PSD", "vesicles", "axon", "boundaries", "mitochondrial boundaries"]

    if data is None:
        data = "2022_11_16"

    if CNN_name is None:
        CNN_name = [
                    "real_data_256_full_unet_6_num_class_27_slices",
                    "real_data_256_tiny_unet_5_num_class_27_slices",
                    "real_data_256_tiny_unet_5_num_class_27_slices_AdamW",
                    "real_data_256_tiny_unet_5_num_class_27_slices_AdamW_v2",
                    "real_data_256_tiny_unet_5_num_class_27_slices_AdamW_v3",
                    "real_data_256_tiny_unet_5_num_class_27_slices_v3",
                    "real_data_256_tiny_unet_6_num_class_27_slices",
                    "real_data_256_tiny_unet_6_num_class_27_slices_AdamW",
                    "real_data_256_tiny_unet_6_num_class_27_slices_AdamW_v2",
                    "real_data_256_tiny_unet_6_num_class_27_slices_AdamW_v3",
                    "real_data_256_tiny_unet_6_num_class_27_slices_no_noise_no_zoom",
                    "real_data_256_tiny_unet_6_num_class_27_slices_v3",
                    "real_data_256_unet_6_num_class_27_slices_no_noise_no_zoom_v2",
                    "real_data_256_unet_6_num_class_27_slices_no_noise_no_zoom_v3_new_gen",
                    "real_data_256_unet_6_num_class_27_slices_no_noise_no_zoom_v3_new_gen_adamW",
                    "real_data_256_unet_6_num_class_27_slices_no_noise_no_zoom_v3_new_gen_adamW_lr0001",
                    "real_data_256_unet_6_num_class_27_slices_no_noise_no_zoom_v3_new_gen_adamW_lr001",
                    "real_data_256_unet_6_num_class_27_slices_no_noise_no_zoom_v3_new_gen_adamW_lr001_v2",
                    "real_data_256_unet_6_num_class_27_slices_no_noise_no_zoom_v3_new_gen_novorad"                      
                   ]

    if list_CNN_num_class is None:
        list_CNN_num_class = [
                              6,
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
                              6,
                              6,
                              6,
                              6,
                              6,
                              6,
                              6,
                              6,
                              ]

    result_CNN_dir = []
    
    for i in range(len(list_CNN_num_class)):
        save_name = "data/result/" + data + "/" + str(list_CNN_num_class[i]) + "_class/" + CNN_name[i]
        result_CNN_dir.append(save_name)

    result_CNN_json_name = []
    
    for i in range(len(list_CNN_num_class)):
        CNN_json_name = "CNN_" + str(list_CNN_num_class[i])
        result_CNN_json_name.append(CNN_json_name)


    save_name_and_dice_result = []

    for i in range(len(list_CNN_num_class)):
    
        save_one_model_info = [CNN_name[i]]

        num_class = list_CNN_num_class[i]
        print(result_CNN_json_name[i])

        print("class\metrics", "jaccard", "dice")
        str_tabel = ""
        for one_str in ["class\metrics", "jaccard", "dice"]:
            str_tabel += one_str + " "
        str_tabel += '\n'

        index_name = 0
        for index_label_name in range(num_class):
            list_name = os.listdir(os.path.join(result_CNN_dir[i] + "_" + str(overlap), mask_name_label_list[index_label_name]))
            json_temp = []
            
            list_for_batch_true = []
            list_for_batch_pred = []
            
            for num, testName in enumerate(list_name):
                print(num + 1, "image is ", len(list_name))

                dir_etal = os.path.join("data", "original data/testing", mask_name_label_list[index_label_name], deleteZero_and_predict_mask(testName))

                etal = io.imread(dir_etal, as_gray=True)
                etal = to_0_255_format_img(etal)
                if (etal.size == 0):
                    print("error etal")

                test_img_name = "predict_" + testName

                test_img_dir = os.path.join(os.path.join(result_CNN_dir[i]+ "_" + str(overlap), mask_name_label_list[index_label_name]), testName)

                img = io.imread(test_img_dir, as_gray=True)

                img = to_0_255_format_img(img)

                if (img.size == 0):
                    print("error img")

                ret, bin_true = cv2.threshold(etal, 128, 255, 0)
                ret, bin_img_true = cv2.threshold(img, 128, 255, 0)

                # print(img)
                # print(etal)
                # viewImage(bin_true,"etal")
                # viewImage(bin_img_true,"img")

                y_true = bin_true.ravel()
                y_pred = bin_img_true.ravel()

                list_for_batch_true.append(y_true)
                list_for_batch_pred.append(y_pred)
            
            y_true = np.array(list_for_batch_true)
            y_pred = np.array(list_for_batch_pred)
            
            rez = jaccard(y_true, y_pred)
            rez2 = dice(y_true, y_pred)
            
            index_name += 1

            len_str = [mask_name_label_list[index_label_name].replace(' ', '_'), rez, rez2]
            save_one_model_info.append(len_str)
            for one_str in len_str:
                str_tabel += str(one_str) + " "
     
        save_name_and_dice_result.append(save_one_model_info)

    for save_one_model_info in save_name_and_dice_result:
        model_name = save_one_model_info[0]
        len_classes = len(save_one_model_info) - 1

        len_name = len(model_name)
            
        num_tab = (50 - len_name) // 4
        
        str_tab = ''
        for t in range(num_tab):
            str_tab += '\t'
       
        str_dice = ''
        
        for c in range(len_classes):
            str_dice +=  " " + str(save_one_model_info[c+1][2])

        print (model_name + str_tab + str_dice)
        
    all_names = 'class/name'
    for i in range(len(save_name_and_dice_result)):
        all_names += save_name_and_dice_result[i][0] + " "
    print(all_names)
    
    for k in range(6):
        class_info = "class_" + str(k) + ": "
        for i in range(len(save_name_and_dice_result)):
            if len(save_name_and_dice_result[i]) - 1 > k:
                class_info += str(save_name_and_dice_result[i][k+1][2]) + " "
            else:
                class_info += " "
        
        print(class_info) 


if __name__ == "__main__":
    TestsMetricDir()