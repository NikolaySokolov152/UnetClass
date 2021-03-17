from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
from keras.utils.np_utils import to_categorical

import cv2
#rgb
any                 = [192, 192, 192]   #wtite-gray
borders             = [0,0,255]         #blue
mitochondria        = [0,255,0]         #green
mitochondria_borders= [0,128,255]
PSD                 = [192,192,64]      #yellow
vesicles            = [255,0,0]         #read

COLOR_DICT = np.array([any, mitochondria, mitochondria_borders, PSD, vesicles,borders])

def view_mask(img,mask ):
    print("img1", img.shape)
    img = img[0,:,:,:] * 255
    img = img.astype(np.uint8)
    print("img2", img.shape)
    cv2.imshow("img", img)
    print("mask",mask.shape)
    num_class = mask.shape[3]
    print(num_class)
    mask_view = mask[0,:,:,:]
    print("mask_view", mask_view.shape)
    img_mask = np.zeros(mask_view.shape[0:2] + (3,))
    print(mask_view[0,0])
    for i in range(num_class):
        img_mask[mask_view[:,:,i] == 1] = COLOR_DICT[i]
    img_mask = img_mask.astype(np.uint8)
    cv2.imshow("mask", img_mask)
    cv2.waitKey()

def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img/255.
        mask = mask[:,:,:,0]
#        new_mask = np.zeros(mask.shape + (num_class,))
#        for i in range(num_class):
#            #new_mask[index_mask] = 1]
#            new_mask[mask == i+1,i] = 1
#       new_mask = np.reshape(new_mask, (new_mask.shape[0],new_mask.shape[1],new_mask.shape[2], new_mask.shape[3]))
        #из-за 0 метки появляется + 1 класс, если правильно поправить код сверху, то он должен работать правильнее
        mask = to_categorical(mask,num_class)
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    #view_mask(img,mask)
    return (img,mask)

def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)

    print(image_generator.class_indices)
    print(mask_generator.class_indices)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = (adjustData(img, mask, flag_multi_class, num_class))
        yield (img,mask)


def testGenerator(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img

def testGenerator2(test_path, name_list = [], num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for img_name in sorted(os.listdir(test_path),key = len)[0:num_image]:
        name_list.append(img_name)
        img = io.imread(os.path.join(test_path, img_name),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img


def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr


def labelVisualize(num_class, trust_percentage, color_dict,img):

#    print(img.shape)
#    for i in range(num_class):
#        print(str(i)+":",  img[:, :, i].max(), " ",img[:, :, i].min() )

    img_out = np.zeros(img.shape[0:2] + (3,))
#    print(img_out.shape)

    for i in range(num_class):
        img_out[img[:,:,i] > trust_percentage] = color_dict[i]
    return img_out/255


def saveResult(save_path,npyfile, trust_percentage = 0.9 ,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class, trust_percentage, COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)

def saveResult2(save_path,npyfile, namelist, trust_percentage = 0.9 ,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class, trust_percentage, COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"predict_"+namelist[i]),img)