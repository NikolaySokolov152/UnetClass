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

name_list = []
mask_list = []

out_dir = "cutting data"

size_data_arr = [256]
size_step_arr = [128]

for i in range(len(size_data_arr)):
	size_data = size_data_arr[i]
	size_step = size_step_arr[i]

	if not os.path.isdir(out_dir):
		print("создаю out_dir:" + out_dir)
		os.makedirs(out_dir)

	dir_input_img = "original/"

	for img_name in os.listdir(dir_input_img):
		count = 0
		if is_Img(os.path.join(dir_input_img, img_name)):
				img = cv2.imread(os.path.join(dir_input_img, img_name), 0)

				if len(img.shape) == 3:
					img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
				low_quality_img = cv2.medianBlur(img, 9)
				#img = agcwd(img)
				h,w = img.shape[0:2]
				
				if not os.path.isdir(out_dir+"/output"):
					print("создаю out_dir:" + "output")
					os.makedirs(out_dir+"/output")
				
				
				if not os.path.isdir(out_dir+"/low_quality"):
					print("создаю out_dir:" + "low_quality")
					os.mkdir(out_dir+"/low_quality")
                

				for start_y in range(0,h, size_step):
					if (h - start_y < size_data):
						continue
					for start_x in range(0,w, size_step):
						if (w - start_x < size_data):
							continue
						cutting_img = img[start_y:start_y+size_data, start_x:start_x+size_data]									
						cv2.imwrite(out_dir + "/output/" + img_name + "_" + str(size_data) +"_" + str(size_step) +"_" +str(count)+".png", cutting_img)
						
						cutting_low_quality = low_quality_img[start_y:start_y+size_data, start_x:start_x+size_data]
						cv2.imwrite(out_dir + "/low_quality/" + img_name + "_" + str(size_data) +"_" + str(size_step) +"_" +str(count)+".png", cutting_low_quality)

						count+=1
		else:
			continue
