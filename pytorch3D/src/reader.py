import cv2
import numpy as np
import skimage.io as io
from skimage import color
import nibabel as nib

AVAILABLE_IMG_TYPE = ('.png', '.jpg', '.jpeg')
nii_type = (".nii", ".nii.gz")

def read_img(path, as_gray=False):
    img = io.imread(path)
    # Убрать альфа канал если он есть (в этом случае колличество каналов четное)
    if len(img.shape)==2:
        return img
    else:
        if img.shape[2]%2 == 1:
            no_alpha_img = img
        else:
            no_alpha_img = img[:,:,:(img.shape[2]-1)]

        if as_gray and img.shape[2] == 3:
            return color.rgb2gray(no_alpha_img)
        else:
            return no_alpha_img.squeeze(-1)

def reader_rgb(img_path):
    image = read_img(img_path)
    return image.astype(np.float32)

def reader_hsv(img_path):
    img = read_img(img_path)
    image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)
    return image.astype(np.float32)

def reader_gray(img_path):
    img = read_img(img_path, as_gray=True)
    image = np.expand_dims(img, -1)
    return image.astype(np.float32)

def read_nii(path):
    nifti_img = nib.load(path)
    # Get the image data as a NumPy array
    image_data = nifti_img.get_fdata()
    return image_data.astype(np.float32)

def get_img_loader(mode):
    if mode == "rgb":
        return reader_rgb
    elif mode == "hsv":
        return reader_hsv
    elif mode == "gray":
        return reader_gray
    else:
        raise Exception(f"Don't known image read mode '{mode}' ")
