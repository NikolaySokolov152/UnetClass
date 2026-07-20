#Splitter

import cv2
import numpy.random as random
import numpy as np
import os
import time

import skimage.io as io

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

file_dir_arr = ["axon", "mitochondria", "PSD", "vesicles", "boundaries","mitochondrial boundaries"]

name_list = []

mask_list = []

out_dir = "cutting data small"

size_data_arr = [256]
size_step_arr = [256]


for i in range(len(size_data_arr)):
	size_data = size_data_arr[i]
	size_step = size_step_arr[i]

	if not os.path.isdir(out_dir):
		print("создаю out_dir:" + out_dir)
		os.makedirs(out_dir)

	dir_input_img = "original data/original/"
	dir_input_mask ="original data/"

	for img_name in os.listdir(dir_input_img):
		count = 0
		if is_Img(os.path.join(dir_input_img, img_name)):
				img = io.imread(os.path.join(dir_input_img, img_name))
				if len(img.shape) == 3:
					img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				#img = agcwd(img)
				h,w = img.shape[0:2]
				
				if not os.path.isdir(out_dir+"/original"):
					print("создаю out_dir:" + "original")
					os.makedirs(out_dir+"/original")
				
				
				for start_y in range(0,h, size_step):
					if (h - start_y < size_data):
						continue
					for start_x in range(0,w, size_step):
						if (w - start_x < size_data):
							continue
						cutting_img = img[start_y:start_y+size_data, start_x:start_x+size_data]
											
						cv2.imwrite(out_dir + "/original/" + img_name + "_" + str(size_data) +"_" + str(size_step) +"_" +str(count)+".png", cutting_img)
						count+=1
		else:
			continue
		

	for i,dir_name in enumerate(file_dir_arr):
		for img_name in os.listdir(dir_input_mask + dir_name):
			if is_Img(os.path.join(dir_input_mask + dir_name, img_name)):
				
				img = cv2.imread(os.path.join(dir_input_mask +dir_name, img_name), 0)
					
				img[img < 128] = 0
				img[img > 127] = 255 
				
				if name_list.count(img_name) == 0:
					name_list.append(img_name)
					mask_list.append(np.zeros((len(file_dir_arr),)+ img.shape, np.uint8))
					
				index = name_list.index(img_name)
				mask_list[index][i] = img			
			else:
				continue
				
	print(name_list)
		
	for index, mask_stack in enumerate(mask_list):
		count = 0
		for i,dir_name in enumerate(file_dir_arr):
			local_count = count
			mask_write = mask_stack[i]
			
			h,w = mask_write.shape[0:2]
		
			if not os.path.isdir(out_dir+"/"+dir_name):
				print("создаю out_dir:" + "mask")
				os.makedirs(out_dir+"/"+dir_name )
			
			for start_y in range(0,h, size_step):
				if (h - start_y < size_data):
					continue
				for start_x in range(0,w, size_step):
					if (w - start_x < size_data):
						continue
					cutting_mask = mask_write[start_y:start_y+size_data, start_x:start_x+size_data]
										
					cv2.imwrite(out_dir+"/"+dir_name +"/" + name_list[index] + "_" + str(size_data) +"_" + str(size_step) +"_" +str(local_count)+".png", cutting_mask)
					local_count+=1
		