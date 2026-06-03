#from model import *
from dataGenerator import DataGenerator, DataGeneratorReaderAll, SaveData, TransformData, InfoDirData
import cv2
import numpy as np

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def to_0_255_format_img(in_img):
    max_val = in_img[:,:].max()
    if max_val <= 1:
       out_img = np.round(in_img * 255)
       return out_img.astype(np.uint8)
    else:
        return in_img


num_class = 6

data_gen_args = dict(rotation_range= 15,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     vertical_flip=True,
                     noise_limit = 5,
                     fill_mode= 0) # cv2.BORDER_CONSTANT = 0

'''
dir_data = InfoDirData(dir_img_name = "G:/Data/my_work/work/human image/unet train/origin",
                       dir_mask_name = "G:/Data/my_work/work/human image/unet train",
                       delete_mask_prefix = '')

#берутся первые классы из списка
mask_name_label_list = ["human_distortion", "human"]
'''
#dir_data = InfoDirData(dir_img_name = "data/train/origin",
#                       dir_mask_name = "data/train/",
#                       add_mask_prefix = 'mask_')

#берутся первые классы из списка
#mask_name_label_list = ["wrong_color", "noise_line"]


#берутся первые классы из списка
mask_name_label_list = ["mitochondria", "PSD", "vesicles", "axon", "boundaries", "mitochondrial_boundaries"]

dir_data = InfoDirData(dir_img_name = "C:/Users/Sokol-PC/Synthetics/Synthetics python/dataset/synthetic_dataset/original",
                       dir_mask_name = "C:/Users/Sokol-PC/Synthetics/Synthetics python/dataset/synthetic_dataset/",
                       add_mask_prefix = '')

#dir_data = InfoDirData(dir_img_name = "G:/Data/Unet_multiclass/data/cutting data/original/",
#                       dir_mask_name = "G:/Data/Unet_multiclass/data/cutting data/",
#                       add_mask_prefix = '')

transform_data = TransformData(color_mode_img='gray',
                               mode_mask='separated',
                               target_size=(256, 256),
                               batch_size=3)

save_inform = SaveData(save_to_dir=None,
                       save_prefix_image="image_",
                       save_prefix_mask="mask_")

#try:
#myGen = DataGenerator(dir_data = dir_data,
myGen = DataGeneratorReaderAll(dir_data = dir_data,
                      num_classes = num_class,
                      mode = "train",
                      aug_dict = data_gen_args,
                      list_class_name = mask_name_label_list,
                      augment = True,
                      tailing = False,
                      shuffle = True,
                      seed = 1,
                      subsampling = "random",
                      transform_data = transform_data,
                      save_inform = save_inform,
                      share_validat = 0.2)
#except Exception as e: print(e)

count = 0

print (len(myGen))
print (len(myGen.gen_train))

print (len(myGen.gen_valid))
size_train =len(myGen.gen_train)

gen_train = myGen.gen_train

print(type(myGen.gen_train))
'''
for rlrm in myGen.gen_train:
    
    print(type(rlrm))
    X,y = rlrm
    
    print(type(X), type(y))
    
    print(X.shape,y.shape)
'''

print("\ntrain\n")

for i in range(2 * size_train):
    x, y = gen_train[i]
    print(x.shape, " ", y.shape)

    for i in range(x.shape[0]):
        #print(str("x")+":",  x[i].max(), " ",x[i].min())
        #print(str("y")+":",  y[i].max(), " ",y[i].min())

        cv2.imshow("test X"+str(i),  x[i])
        
        
        cv2.imshow("test Y"+str(i)+"wrong",  to_0_255_format_img(y[i,:,:,0]))
        cv2.imshow("test Y"+str(i)+"nois",  to_0_255_format_img(y[i,:,:,1]))
        #for j in range(y.shape[-1]):
        #    cv2.imshow("test Y_" + mask_name_label_list[j],  y[i][:,:,j])
    cv2.waitKey()
    count+=1


print("\nvalid\n")

for i in range(2 * len(myGen.gen_valid)):
    x, y = myGen.gen_valid[i]
    print(x.shape, " ", y.shape)

    for i in range(x.shape[0]):
        #print(str("x")+":",  x[i].max(), " ",x[i].min())
        #print(str("y")+":",  y[i].max(), " ",y[i].min())

        cv2.imshow("test X"+str(i),  x[i])
        
        
        cv2.imshow("test Y"+str(i)+"wrong",  to_0_255_format_img(y[i,:,:,0]))
        cv2.imshow("test Y"+str(i)+"nois",  to_0_255_format_img(y[i,:,:,1]))
        #for j in range(y.shape[-1]):
        #    cv2.imshow("test Y_" + mask_name_label_list[j],  y[i][:,:,j])
    cv2.waitKey()
    count+=1
    
myGen.on_epoch_end()

print("\ntrain\n")

for i in range(2 * size_train):
    x, y = gen_train[i]
    print(x.shape, " ", y.shape)

    for i in range(x.shape[0]):
        #print(str("x")+":",  x[i].max(), " ",x[i].min())
        #print(str("y")+":",  y[i].max(), " ",y[i].min())

        cv2.imshow("test X"+str(i),  x[i])
        
        
        cv2.imshow("test Y"+str(i)+"wrong",  to_0_255_format_img(y[i,:,:,0]))
        cv2.imshow("test Y"+str(i)+"nois",  to_0_255_format_img(y[i,:,:,1]))
        #for j in range(y.shape[-1]):
        #    cv2.imshow("test Y_" + mask_name_label_list[j],  y[i][:,:,j])
    cv2.waitKey()
    count+=1


print("\nvalid\n")
for i in range(2 * len(myGen.gen_valid)):
    x, y = myGen.gen_valid[i]
    print(x.shape, " ", y.shape)

    for i in range(x.shape[0]):
        #print(str("x")+":",  x[i].max(), " ",x[i].min())
        #print(str("y")+":",  y[i].max(), " ",y[i].min())

        cv2.imshow("test X"+str(i),  x[i])
        
        
        cv2.imshow("test Y"+str(i)+"wrong",  to_0_255_format_img(y[i,:,:,0]))
        cv2.imshow("test Y"+str(i)+"nois",  to_0_255_format_img(y[i,:,:,1]))
        #for j in range(y.shape[-1]):
        #    cv2.imshow("test Y_" + mask_name_label_list[j],  y[i][:,:,j])
    cv2.waitKey()
    count+=1



myGen.on_epoch_end()
myGen.on_epoch_end()
myGen.on_epoch_end()
myGen.on_epoch_end()
