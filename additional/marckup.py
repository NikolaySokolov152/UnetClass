import cv2
import numpy.random as random
import numpy as np
import os
import time

#borders
#mitochondria
#mitochondria borders
#PSD
#vesicles

def is_Img(name):
	img_type = ('.png', '.jpg', '.jpeg')
	if name.endswith((img_type)):
		return True
	else:
		return False

file_dir_arr = ["mitochondria","mitochondria borders", "PSD", "vesicles", "borders"]

name_list = []

mask_list = []

classes = 1
for dir_name in file_dir_arr:
	for img_name in os.listdir(dir_name):
		if is_Img(os.path.join(dir_name, img_name)):
			
			img = cv2.imread(os.path.join(dir_name, img_name), 0)
			if name_list.count(img_name) == 0:
				name_list.append(img_name)
				
				img[img < 127] = 0
				img[img > 127] = classes 
				
				mask_list.append(img)
			
			else:
				index = name_list.index(img_name)
				print(dir_name, img_name ,index)
				
				change_img = mask_list[index]
				change_img[img > 127] = classes
				mask_list[index] = change_img
		else:
			continue
			
	classes+=1
	
print(classes)
print(name_list)

def line_img(img):
	min = img.min()
	max = img.max()
	clone = img.copy()
	clone = (clone-min) / (max-min) * 255
	clone = clone.astype(np.uint8)
	return clone

for i in range(0,5):
	mask_write = mask_list[i]
	cv2.imshow("mask_"+str(i)+".png", line_img(mask_write))
	cv2.imwrite("mask_"+str(i)+".png", mask_write)
	cv2.waitKey()
