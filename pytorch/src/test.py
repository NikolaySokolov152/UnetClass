import sys
if not __name__ == "__main__":
    sys.path.append("src/")
from splitImages import *
from models import *
import torch

import cv2
import numpy as np
import skimage.io as io
import time
import tqdm


EPSILON = 1e-7
########################################################## разобраться с чтеним без альфа канала

def to_0_1_format_img(in_img):
    max_val = in_img[:,:].max()
    if max_val <= 1:
        return in_img
    else:
        out_img = in_img.astype(np.float32) / 255
        return out_img

class tiledGen():
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        item = self.data[index]
        img = np.reshape(item, (1,) + item.shape + (1,))
        torch_img = torch.from_numpy(np.array(img)).type(torch.FloatTensor).permute(0, 3, 1, 2)

        return torch_img

    def __len__(self):
        return len(self.data)

def glit_mask(tiled_masks, num_class, out_size, tile_info, overlap = 64):
    masks = []
    for i_class in range(num_class):
        pic = tiled_masks.take(i_class, axis=-1)
        i_mask = glit_image(pic, out_size, tile_info, overlap)
        #print(result_class.shape)
        masks.append(i_mask)

    #print(masks[0].shape)
    union_arr = np.zeros(out_size + (num_class,), np.uint8)
    for i_class in range(num_class):
        union_arr[:,:,i_class] = masks[i_class]
    #print(union_arr.shape)

    return np.reshape(union_arr, (1,) + union_arr.shape)

def saveResultMask(save_path, npyfile, namelist, num_class = 2 , classnames=None):
    for i,item in enumerate(npyfile):
        for class_index in range(num_class):
            out_dir = os.path.join(save_path, classnames[class_index] if classnames is not None else str(class_index))
            if not os.path.isdir(out_dir):
                if os.name == 'nt':  # for Windows
                    print("создаю out_dir:" + out_dir.replace(u"\\\\?\\"+os.getcwd()+"\\", ""))
                else:
                    print("создаю out_dir:" + out_dir.replace(os.getcwd() + "\\", ""))
                os.makedirs(out_dir)

            if (os.path.isfile(os.path.join(out_dir, "predict_" + namelist[i]))):
                os.remove(os.path.join(out_dir, "predict_" + namelist[i]))

            io.imsave(os.path.join(out_dir, "predict_" + namelist[i]), item[:,:,class_index], check_contrast=False)

def predictModel(model, data, device, last_activation, eps = EPSILON):
    result = []
    model.eval()
    with torch.no_grad():
        #time.sleep(0.2)  # чтобы tqdm не печатал вперед print
        #tqdm_test_loop = tqdm.tqdm(data, file=sys.stdout, desc="\tSlice", colour="GREEN")
        for epoch_valid_iteration, inputs in enumerate(data):
            inputs = inputs.to(device)
            outputs = model(inputs)
            # ADD LAST ACTIVATION
            outputs = globals()[last_activation](outputs, eps)

            result.append(outputs.detach().cpu().permute(0, 2, 3, 1).numpy()[0])
    return np.array(result)

def test_tiled(model_path, num_class, save_mask_dir, last_activation = None, dataset={'filenames': None, "filepath": "data/test", "classnames": None},
               tiled_data={"size":256, "overlap":64, "unique_area":0}, save_dir = None):
    filenames = dataset["filenames"]
    filepath = dataset["filepath"]
    classnames = dataset["classnames"]

    if len(filenames) == 0:
        raise Exception(f"No image to predict")

    size = tiled_data["size"]
    overlap = tiled_data["overlap"]
    unique_area = tiled_data["unique_area"]

    model = torch.load(model_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    last_activation = "sigmoid_activation" if last_activation is None else last_activation
    model.to(device)

    ret_images = []
    print(last_activation)
    time.sleep(0.2)

    if save_mask_dir is not None:
        save_mask_dir = os.path.abspath(save_mask_dir)

        if os.name == 'nt': # for Windows
            if save_mask_dir.startswith(u"\\\\"):
                save_mask_dir = u"\\\\?\\UNC\\" + save_mask_dir[2:]
            else:
                save_mask_dir = u"\\\\?\\" + save_mask_dir

    slices_tqdm = tqdm.tqdm(filenames, file=sys.stdout, desc="Test")
    for img_name in slices_tqdm:
        ##########################################################
        #img = io.imread(os.path.join(filepath, img_name))
        # io открывает с альфа каналом, поэтому всего может быть и 2 и 4 канала (1, 2, 3 ,4)
        # поэтому пока что открываю всё в сером !
        img = cv2.imread(os.path.join(filepath, img_name), cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise Exception(f"No open predict image '{img_name}'")

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        img = to_0_1_format_img(img)

        # delete suffix (.png, .jpg)
        tiled_name = img_name.split('.')[0]
        tiled_arr, tile_info = split_image(img, tiled_name, save_dir, size, overlap, unique_area)

        img_generator = tiledGen(tiled_arr)

        results = predictModel(model, img_generator, device, last_activation)

        res_img = glit_mask(results, num_class, img.shape, tile_info, overlap)
        # print("glit_mask", res_img.shape)

        ################################################################################################################ вспомнить почему нужна единичная ось в начале
        ################################### нужно для корректной работы универсальной функции сохранения (подумать нужно ли это вообще)
        ret_images.append((img_name, res_img[0]))
        if save_mask_dir is not None:
            saveResultMask(save_mask_dir, res_img, [img_name], num_class=num_class, classnames=classnames)

    return ret_images