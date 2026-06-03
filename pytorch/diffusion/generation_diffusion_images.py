import torch
import numpy as np
import skimage.io as io
import sys
import tqdm
import os
import cv2

#if __name__ == "__main__":
sys.path.append("../")
sys.path.append("../src/")
   
from models import *
model_path =  "epoch_model_by_config_test_&_tiny_unet_v3_!.pt"
#model_path =  "model_by_config_test_&_tiny_unet_v3_!.pt"

if os.path.isfile(model_path):
    print("File is found")
else:
    print("ERROR file founded")

model = torch.load(model_path)
last_activation="sigmoid_activation"


EPSILON = 1e-7


num_steps_denoise = 350

device = "cuda"
out_dir = "result"

#def predictModel(model, device, last_activation, eps = EPSILON):
result = []
model.to(device)
model.eval()
with torch.no_grad():
    new_image = torch.randn((1,1, 256,256), device=device)
    print(new_image.shape)
    
    result.append(new_image.cpu().permute(0, 2, 3, 1).numpy())
    for i in tqdm.tqdm(range(num_steps_denoise), file=sys.stdout, desc="Test"):
        outputs = model(new_image)
        new_image_denoise=globals()[last_activation](outputs, EPSILON)
        #new_image_denoise=outputs
        result.append(new_image_denoise.cpu().permute(0, 2, 3, 1).numpy())
        new_image=new_image_denoise

for i, image in enumerate(result):
    if not os.path.isdir(out_dir):
        print("создаю out_dir:" + out_dir)
        os.makedirs(out_dir)
    #print(image.shape)
    
    #one_image = to_0_255_format_img(image[0,:,:,0])
    #print(one_image)
    #cv2.imshow(f"{i} image", image[0])
    #cv2.imshow(f"{i} image 2", one_image)
    #cv2.waitKey()
    
    #print(one_image.shape, one_image.dtype)
    
        
    #io.imsave(os.path.join(out_dir, f"predict_{i}_step_denoise.png"), one_image , check_contrast=False)
    cv2.imwrite(os.path.join(out_dir, f"predict_{i}_step_denoise.png"), image[0]*255) 

