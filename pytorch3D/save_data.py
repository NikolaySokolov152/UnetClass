import os
import skimage.io as io

def saveDataframeAsImgs(save_path, filename, dataframe, classes_list=None):
    print("save dataframe", save_path)
    if classes_list is None:
        for z, img in enumerate(dataframe): # shape(d, h, w, c)
            io.imsave(os.path.join(save_path, filename + f"_{z}.png"), img[:,:,0], check_contrast=False)
            
    else:
        for i in range(len(classes_list)):
            now_class_name = classes_list[i]
            
            if not os.path.isdir(os.path.join(save_path, now_class_name)):
                print("create dir", os.path.join(save_path, now_class_name))
                os.mkdir(os.path.join(save_path, now_class_name))

            for z, img in enumerate(dataframe): # shape(d, h, w, c)
                io.imsave(os.path.join(save_path, now_class_name, filename + f"_{z}.png"), img[:,:,i], check_contrast=False)