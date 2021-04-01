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

file_dir_arr = ["mask mitohondry 2"]

classes = 1
for dir_name in file_dir_arr:
	count = 0
	for img_name in os.listdir(dir_name):
	
		if is_Img(os.path.join(dir_name, img_name)):
			
			img = cv2.imread(os.path.join(dir_name, img_name), 0)
						
			img[img < 127] = 0
			img[img > 127] = classes 
			
			out_dir_name = "label_mask"

			if not os.path.isdir(out_dir_name + "/" +dir_name):
				print("создаю " + dir_name)
				os.makedirs(out_dir_name + "/" +dir_name)
			
			cv2.imwrite(out_dir_name + "/" + dir_name + "/"+str(count)+".png", img)
			cv2.waitKey()

		else:
			continue
			
		count +=1
	classes +=1
	
