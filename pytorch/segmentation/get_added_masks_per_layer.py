import skimage.io as io
from skimage import color
import os
import numpy as np
import cv2


def to_0_255_format_img(in_img):
    max_val = in_img[:, :].max()
    if max_val <= 1:
        out_img = np.round(in_img * 255)
        return out_img.astype(np.uint8)
    else:
        return in_img

path_test_layers = "G:/Data/Unet_multiclass/data/original data/testing/original"
path_test_masks = "G:/Data/Unet_multiclass/data/original data/testing"
images_names = [name for name in os.listdir(path_test_layers) if name.endswith(".png")]

color_orig_mask = (0, 0, 255)
color_pred_mask = (255, 0, 0)

alpha_orig = 0.5
alpha_pred = 0.5

classes = ["mitochondria", "PSD", "vesicles", "axon", "boundaries", "mitochondrial boundaries"]

#path_model_result_arr = ["data/result/Test_mix_v2_dataset/6_class"]
#model_names = ["model_by_config_multiclass_mix_v2_6_classes_DiceLossMulticlass_tiny_unet_v3_128"]
#num_classes_arr = [6]

models = [
          "unet",
          "tiny_unet_v3",
          "mobile_unet",
          "Lars76_unet"
         ]
num_classes = [1, 5, 6]
types_datasets = [
                  "real_v2",
                  "sint_v10",
                  "mix_v2"
                  ]
losses = [
          "DiceLossMulticlass",
          ]
overlap_list = [128]

path_model_result_arr = []
model_names = []
num_classes_arr = []
lists_type_dataset = []
for type_dataset in types_datasets:
    for num_class in num_classes:
        for model_name in models:
            for loss in losses:
                for overlap in overlap_list:
                    lists_type_dataset.append(f"Test_{type_dataset}_dataset")
                    path_model_result_arr.append(f"data/result/Test_{type_dataset}_dataset/{num_class}_class")
                    model_names.append(f"model_by_config_multiclass_{type_dataset}_{num_class}_classes_{loss}_{model_name}_{overlap}")
                    num_classes_arr.append(num_class)



for i, model_name in enumerate(model_names):
    path_model_result = path_model_result_arr[i]
    n_classes = num_classes_arr[i]
    dataset_name = lists_type_dataset[i]
    save_path = f"view_data/{dataset_name}/{n_classes}_class/{model_name}"

    for name in images_names:
        print(name)
        layer = cv2.imread(os.path.join(path_test_layers, name))
        for class_name in classes[:n_classes]:
            layer_with_mask = layer.copy().astype(float)


            mask_orig = cv2.imread(os.path.join(path_test_masks, class_name, name), 0)
            mask_pred = cv2.imread(os.path.join(path_model_result, model_name, class_name, "predict_"+name), 0)

            mask_orig[mask_orig>127] = 255
            mask_orig[mask_orig<128] = 0

            mask_pred[mask_pred>127] = 255
            mask_pred[mask_pred<128] = 0

            mask_inter = mask_orig&mask_pred

            mask_orig = mask_orig-mask_inter
            mask_pred = mask_pred-mask_inter

            layer_with_mask[mask_orig == 255] = np.multiply(layer_with_mask[mask_orig == 255], (1 - alpha_orig)) + np.multiply(color_orig_mask, alpha_orig)
            layer_with_mask[mask_pred == 255] = np.multiply(layer_with_mask[mask_pred == 255], (1 - alpha_pred)) + np.multiply(color_pred_mask, alpha_pred)

            layer_with_mask[mask_inter == 255] = np.multiply(layer_with_mask[mask_inter == 255],
                                                            (1 - alpha_orig)) + np.multiply(np.add(color_orig_mask,color_pred_mask), alpha_orig)

            save_result_path = f"{save_path}/{class_name}"

            if not os.path.isdir(save_result_path):
                os.makedirs(save_result_path)

            cv2.imwrite(save_result_path + f"/{name}.png", to_0_255_format_img(layer_with_mask))

            save_result_path_dataset_class =  f"view_data_all/{dataset_name}/{class_name}"
            if not os.path.isdir(save_result_path_dataset_class):
                os.makedirs(save_result_path_dataset_class)

            cv2.imwrite(save_result_path_dataset_class + f"/{name}_{model_name}.png", to_0_255_format_img(layer_with_mask))
