import numpy as np
import os

from PIL import Image
from test_metrics import Dice, Jaccard
from read_fun import sort_filenames, check_sorted
from src.read_data_transform_fun import to_0_1_format_img, to_0_255_format_img


import re
AVAILABLE_IMG_TYPE = ('.png', '.jpg', '.jpeg')

UsemodelName = "3D_Attention_UNet_dropaut_0_num_class_5_sint_only_batch_6_shape_data_80_2026_07_13_01_20_47"
TEST_PREDICT_MULTICLASS_PATH_DATA = f"model_predict/{UsemodelName}"
TEST_ETAL_MITO_PATH_DATA = "D:/Data/datasets/Lucchi++"
TEST_ETAL_MULTICLASS_PATH_DATA = "D:/Projects/UnetClass/pytorch3D/segmentation/data/original data/testing"
number_of_test_classes = 5
classes_etal_list = ["mitochondria", "boundaries", "vesicles", "axon", "PSD"]



mito_test_img_names_all = [name for name in os.listdir(os.path.join(TEST_ETAL_MITO_PATH_DATA, "Test_Out")) if name.endswith((AVAILABLE_IMG_TYPE))]
mito_test_img_names_all = sorted(mito_test_img_names_all, key=len)
print(check_sorted(mito_test_img_names_all))
print(mito_test_img_names_all)

############################################
print("\nполучение эталонных масок митохондрий")
# получение данных из масок
test_etal_mito_maskframe = []
for name in mito_test_img_names_all:
    one_mask_frame = []
    class_name = "Test_Out"
    syn_mask = Image.open(os.path.join(TEST_ETAL_MITO_PATH_DATA, class_name, name)).convert("L")
    one_mask_frame.append(syn_mask)
    syn_mask = np.stack(one_mask_frame)
    syn_mask = to_0_1_format_img(syn_mask)
    test_etal_mito_maskframe.append(syn_mask)

print(len(test_etal_mito_maskframe))
test_etal_mito_maskframe = np.transpose(np.stack(test_etal_mito_maskframe), (0, 2, 3, 1))
print(test_etal_mito_maskframe.shape)


# чтение имеющейся разметки
print("\ntest_multimaskframe")

multi_test_img_names_all = [name for name in os.listdir(os.path.join(TEST_ETAL_MULTICLASS_PATH_DATA, classes_etal_list[0])) if name.endswith((AVAILABLE_IMG_TYPE))]
multi_test_img_names_all = sorted(multi_test_img_names_all, key=len)

print(check_sorted(multi_test_img_names_all))
print(multi_test_img_names_all)



############################################
print("\nполучение эталонных мультиклассовых масок ")
# получение данных из масок
test_etal_multiclass_maskframe = []
for name in multi_test_img_names_all:
    one_mask_frame = []
    for i in range(number_of_test_classes):
        class_name = classes_etal_list[i]
        syn_mask = Image.open(os.path.join(TEST_ETAL_MULTICLASS_PATH_DATA, class_name, name)).convert("L")
        one_mask_frame.append(syn_mask)
    syn_mask = np.stack(one_mask_frame)
    syn_mask = to_0_1_format_img(syn_mask)
    test_etal_multiclass_maskframe.append(syn_mask)

print(len(test_etal_multiclass_maskframe))
test_etal_multiclass_maskframe = np.transpose(np.stack(test_etal_multiclass_maskframe), (0, 2, 3, 1))
print(test_etal_multiclass_maskframe.shape)

mask_indexes_in_stak = []
for filename in multi_test_img_names_all:
    match = re.search(r'(\d+)', filename)
    if match:
        number_str = match.group(1)
        number = int(number_str)
        mask_indexes_in_stak.append(number)

print(mask_indexes_in_stak)



print("\nполучение масок предикта")
# получение данных из масок


classes_predict_list = ["mitohondrion", "membranes", "vesicles", "axon", "psd"]

predict_test_img_names_all = [name for name in os.listdir(os.path.join(TEST_PREDICT_MULTICLASS_PATH_DATA, classes_predict_list[0])) if name.endswith((AVAILABLE_IMG_TYPE))]
predict_test_img_names_all = sorted(predict_test_img_names_all, key=len)
print(check_sorted(predict_test_img_names_all))
print(predict_test_img_names_all)



test_multiclass_predict_maskframe = []
for name in predict_test_img_names_all:
    one_mask_frame = []
    for i in range(number_of_test_classes):
        class_name = classes_predict_list[i]
        syn_mask = Image.open(os.path.join(TEST_PREDICT_MULTICLASS_PATH_DATA, class_name, name)).convert("L")
        one_mask_frame.append(syn_mask)
    syn_mask = np.stack(one_mask_frame)
    syn_mask = to_0_1_format_img(syn_mask)
    test_multiclass_predict_maskframe.append(syn_mask)

print(len(test_multiclass_predict_maskframe))
test_multiclass_predict_maskframe = np.transpose(np.stack(test_multiclass_predict_maskframe), (0, 2, 3, 1))
print(test_multiclass_predict_maskframe.shape)



binary_multiresult = test_multiclass_predict_maskframe.copy()
binary_multiresult[binary_multiresult < 0.5] = 0
binary_multiresult[binary_multiresult > 0] = 1



binary_result = binary_multiresult[:,:,:,0]
etal_val = test_etal_mito_maskframe[:,:,:,0]    

print("Предсказание по Lucchi++")
print(f"Модель {UsemodelName} имеет качество по Dice для класса {classes_predict_list[0]}: {Dice(binary_result, etal_val)}")
print(f"Модель {UsemodelName} имеет качество по IoU  для класса {classes_predict_list[0]}: {Jaccard(binary_result, etal_val)}")



list_predicts_by_index = [binary_multiresult[index] for index in mask_indexes_in_stak]
predict_multiframe = np.stack(list_predicts_by_index)
print(predict_multiframe.shape)


print("\nПредсказание по ITMM")
for n in range(number_of_test_classes):
    binary_result = predict_multiframe[:,:,:,n]
    etal_val = test_etal_multiclass_maskframe[:,:,:,n]    

    print(f"Модель {UsemodelName} имеет качество по Dice для класса {classes_predict_list[n]}: {Dice(binary_result, etal_val)}")
    print(f"Модель {UsemodelName} имеет качество по IoU  для класса {classes_predict_list[n]}: {Jaccard(binary_result, etal_val)}")
    
   