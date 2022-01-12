import cv2
import numpy.random as random
import numpy as np
import os
import time


import skimage.io as io
from AGCWD import*


dir_input_img = "original data/original/"
dir_input_mask = "original data/"

file_dir_arr = ["mitochondria", "axon", "PSD", "vesicles", "boundaries","mitochondrial boundaries"]

n_class = 6

def is_Img(name):
    img_type = ('.png', '.jpg', '.jpeg')
    if name.endswith((img_type)):
        return True
    else:
        return False
        
gist_value = np.zeros((n_class, 256), np.uint64)

        
for img_name in os.listdir(dir_input_img):
    if is_Img(img_name):
        img = cv2.imread(os.path.join(dir_input_img, img_name), 0)
        img = agcwd(img)
        
        cv2.imshow(img_name, img)
        print(img.shape)
        cv2.waitKey()
        
    for i in range(n_class):

        mask = cv2.imread(os.path.join(dir_input_mask, file_dir_arr[i], img_name), 0)

        mask_i = img[mask == 255].ravel()

        for value in mask_i:
            gist_value[i][value] = gist_value[i][value] + 1
        
        

for i in range(n_class):
    np.savetxt("gistogramm agcwd " + file_dir_arr[i] + ".txt", [gist_value[i]], fmt="%d", delimiter=' ')
    