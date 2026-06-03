import cv2
import os
import numpy as np

path_img = "result_t_dataset_6_classes_01_mask/original"

list_name = [name for name in os.listdir(path_img) if name.endswith(".png")]

counter_bad_mean = 0
counter_bad_std = 0
counter_bad_mean_std = 0

lim_mean = [126,164]
lim_std = [22,35]

good_mean_img_list = []
good_std_img_list = []
good_mean_std_img_list = []

mean_list = []
std_list = []

# mean mean dataset: 138.83564813572747
# mean std dataset: 29.684126721380455

for name in list_name:
    img = cv2.imread(os.path.join(path_img, name), 0)

    mean_img = np.mean(img)
    std_img = np.std(img)

    img=img.astype(float)
    img = (img-mean_img)/std_img
    img = (img*29.684126721380455)+138.83564813572747

    img[img>255]=255
    img[img<0]=0
    img = img.astype(np.uint8)    

    mean_list.append(mean_img)
    std_list.append(std_img)

    if lim_mean[0] < mean_img < lim_mean[1]:
        good_mean_img_list.append(img)
    else:
        counter_bad_mean+=1

    if lim_std[0] < std_img < lim_std[1]:
        good_std_img_list.append(img)
    else:
        counter_bad_std+=1

    if lim_mean[0] < mean_img < lim_mean[1] and lim_std[0] < std_img < lim_std[1]:
        good_mean_std_img_list.append(img)
    else:
        counter_bad_mean_std+=1

    cv2.imwrite(f"origin_by_mean_std/{name}", img)


print(f"count images : {len(list_name)}")

print(f" mean mean dataset: {np.mean(mean_list)}") 
print(f" mean std dataset: {np.mean(std_list)}")  

print(f" min mean dataset: {min(mean_list)}")    
print(f" max mean dataset: {max(mean_list)}")    
print(f" min std dataset: {min(std_list)}")     
print(f" max std dataset: {max(std_list)}")     

print(f"counter_bad_mean: {counter_bad_mean}")
print(f"counter_bad_std: {counter_bad_std}")
print(f"counter_bad_mean_std: {counter_bad_mean_std}")