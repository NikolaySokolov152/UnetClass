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

out_dir = "cutting data"

size_data = 256
size_step = 128
start_count = 0

if not os.path.isdir(out_dir):
	print("создаю out_dir:" + out_dir)
	os.makedirs(out_dir)

dir_input_img = "data/original/"
dir_input_mask ="data/"


count = start_count
for img_name in os.listdir(dir_input_img):
	if is_Img(os.path.join(dir_input_img, img_name)):
			img = cv2.imread(os.path.join(dir_input_img, img_name), 0)
			h,w = img.shape[0:2]
			
			
			if not os.path.isdir(out_dir+"/image"):
				print("создаю out_dir:" + "image")
				os.makedirs(out_dir+"/image")
			
			
			for start_y in range(0,h, size_step):
				if (h - start_y < size_data):
					continue
				for start_x in range(0,w, size_step):
					if (w - start_x < size_data):
						continue
					cutting_img = img[start_y:start_y+size_data, start_x:start_x+size_data]
										
					cv2.imwrite(out_dir + "/image/"+str(count)+".png", cutting_img)
					count+=1
	else:
		continue
	


classes = 1

for dir_name in file_dir_arr:
	for img_name in os.listdir(dir_input_mask + dir_name):
		if is_Img(os.path.join(dir_input_mask + dir_name, img_name)):
			
			img = cv2.imread(os.path.join(dir_input_mask +dir_name, img_name), 0)
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
	
count = start_count
	
for i in range(0,5):
	mask_write = mask_list[i]
	
	h,w = mask_write.shape[0:2]
			
			
	if not os.path.isdir(out_dir+"/mask"):
		print("создаю out_dir:" + "mask")
		os.makedirs(out_dir+"/mask")
	
	
	for start_y in range(0,h, size_step):
		if (h - start_y < size_data):
			continue
		for start_x in range(0,w, size_step):
			if (w - start_x < size_data):
				continue
			cutting_mask = mask_write[start_y:start_y+size_data, start_x:start_x+size_data]
								
			cv2.imwrite(out_dir + "/mask/"+str(count)+".png", cutting_mask)
			count+=1
