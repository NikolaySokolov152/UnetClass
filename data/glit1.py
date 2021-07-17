"""
Created on Mon Mar  12 11:21:42 2019

@author: Vasiliev Evgeniy

File for splitting of one big image for small images 
and create big image from small images
 
"""
import os
import cv2
import numpy as np

def split_image_file(filename, mode = 'color', folder = '', newsize = (256, 256)):
    if mode == 'gray':
        m = 0
    elif mode == 'color':
        m = 1
    bigimage = cv2.imread(filename, m)
    bigimagename = os.path.splitext(os.path.basename(filename))[0]
    
    height = bigimage.shape[0]
    width = bigimage.shape[1]
    if bigimage.ndim == 2:
        channels = 1
    else:
        channels = 3
    
    newh = newsize[1]
    neww = newsize[0]
    nx = (width + newsize[0]-1) // newsize[0]
    ny = (height + newsize[1]-1) // newsize[1]
    
    for iy in range(ny):
        for ix in range (nx):
            if channels == 1:
                smallimg_dim = (newh, neww)
            else:
                smallimg_dim = (newh, neww, 3)
            smallimg = bigimage[iy*newh : (iy+1)*newh, 
                                ix*neww : (ix+1)*neww]
            newname = '{}{}_{}_{}.png'.format(folder, bigimagename, iy, ix)
            cv2.imwrite(newname, smallimg)
    
    
def find_big_image_size(folder, prefix, imageslist):
    # Находим размеры маленькой картинки (которая не обрезок справа или снизу)
    im = cv2.imread(folder+imageslist[0])
    imagesize = (im.shape[0], im.shape[1])
    
    # Находим из какого количества сегментов состоит большая картинка    
    names = [i.replace(prefix,'').replace('.','_').split('_') for i in imageslist]
    yy = max([int(i[1]) for i in names])
    xx = max([int(i[2]) for i in names])
    imagenum = (yy+1, xx+1)
    
    # Находим размеры большой картинки
    name = [i for i in imageslist if i.find('{}_{}'.format(yy, xx))>0]
    lastim = cv2.imread(folder+name[0])
    lastimagesize = (lastim.shape[0], lastim.shape[1])
    bigimagesize = (imagesize[0] * yy + lastimagesize[0], 
                    imagesize[1] * xx + lastimagesize[1])
    return bigimagesize, imagenum, imagesize
    
def glue_image_file(folder, name_prefix, newname, mode = 'color'):
    if mode == 'gray':
        m = 0
    elif mode == 'color':
        m = 1
    fileslist = [i for i in os.listdir(folder) if i.find(name_prefix) >= 0]
    
    bigimageshape, imnum, imsize = find_big_image_size(folder, name_prefix,  fileslist)
    bigimage = np.zeros(bigimageshape+(3,), np.uint8)
    
	
    ny = imnum[0]
    nx = imnum[1]
    newh = imsize[0]
    neww = imsize[1]
    
    for iy in range(ny):
        for ix in range(nx):
            smallimname = '{}{}_{}_{}.png'.format(folder, name_prefix, iy, ix)
            smallim = cv2.imread(smallimname, m)
            
            bigimage[iy*newh : iy*newh + smallim.shape[0], 
                     ix*neww : ix*neww + smallim.shape[1]] = smallim
    
    cv2.imwrite(newname, bigimage)

	
def is_Img(name):
	img_type = ('.png', '.jpg', '.jpeg')
	if name.endswith((img_type)):
		return True
	else:
		return False

			
			
def splitt(originalimage, splittedfolder):
	for img_name in os.listdir(originalimage):
		if is_Img(os.path.join(originalimage, img_name)):
			split_image_file(os.path.join(originalimage, img_name), 'gray', splittedfolder, (256, 256))# Ширина и высота
		else:
			continue
	
glit_index_name = ["0003"]
	
def glit(splittedfolder,glitoutput, count_img, pref_name = ""):
	for i in range(count_img):
		prifix_name = pref_name + glit_index_name[i]
		processedimage = glitoutput + str(i) + '_glit.png'
		glue_image_file(splittedfolder, prifix_name, processedimage, 'color')

mask_name_label_list = ["mitochondria", "PSD", "vesicles", "axon", "boundaries", "mitochondrial boundaries"]

for i in range(1):
    originalimage = os.path.join('result/', mask_name_label_list[i]) + "/"
    glitoutput = os.path.join('result/converted/', mask_name_label_list[i])
    splittedfolder = os.path.join('result/', mask_name_label_list[i]) + "/"
    
    if not os.path.isdir(glitoutput):
      print("создаю out_dir:" + glitoutput)
      os.makedirs(glitoutput)

    glitoutput += "/"
    #splitt(originalimage, splittedfolder)
    glit(splittedfolder, glitoutput, 1, "predict_training")