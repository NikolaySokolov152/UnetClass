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

device = "cuda"

batch = 16
n_pic = 1000//batch+1
num_steps_denoise = 700

channel_names = ["original", "mitochondria", "PSD", "vesicles", "axon", "boundaries", "mitochondrial boundaries"]

#n_pic = 1

model_paths = [
            #"2024_03_25/model_by_config_diffusion_tiny_unet_v3_MSELoss_6_classes_mean_std.pt",
            #"2024_03_25/model_by_config_diffusion4_tiny_unet_v3_MSELoss_1_class_mean_std.pt",
            #"2024_03_25/model_by_config_diffusion2_tiny_unet_v3_MSELoss_5_classses_mean_std.pt",
            "2024_03_25/model_by_config_diffusion3_tiny_unet_v3_MSELoss_100_slices_mean_std.pt"            
            ]


add_names_list=[
            #"_dataset_6_classes_01_mask",
            #"_dataset_1_classes_01_mask",
            #"_dataset_5_classes_01_mask",
            "_dataset_1_class_100_slices_01_mask"
          ]

n_cannels_list = [
                #7,
                #2,
                #6,
                2
                ]






betas = torch.linspace(0.0001, 0.02, num_steps_denoise)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
def img_to_uint(img):
    ret_img = (img*29.684126721380455)+138.83564813572747
    ret_img[ret_img>255]=255
    ret_img[ret_img<0]=0
    return ret_img.astype(np.uint8)


def mask_to_uint(img):
    ret_img=img*255
    ret_img[ret_img>255]=255
    ret_img[ret_img<0]=0
    return ret_img.astype(np.uint8)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def p_sample(model, x, t, t_index):

    betas_t = extract(betas, t, x.shape).to(device)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    ).to(device)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape).to(device)
    
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    res = model(x, t)[0]
    
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * res/sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape).to(device)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise 


for i, model_path in enumerate(model_paths):
    if os.path.isfile(model_path):
        print(f"File {model_path} is found")
    else:
        print("ERROR file founded")
    model = torch.load(model_path)
    model.to(device)
    model.eval()

    n_cannels = n_cannels_list[i]
    add_name = add_names_list[i]

    out_dir = f"result_t{add_name}"

    result = []

    with torch.no_grad():
        img_index=0
        if not os.path.isdir(out_dir):
            print("создаю out_dir:" + out_dir)
            os.makedirs(out_dir)

        
        for iter in tqdm.tqdm(range(n_pic), desc='sampling loop time step', total=n_pic):
            print(f"stage_gen: {iter}", flush=True)
            img = torch.randn((batch, n_cannels, 256,256), device=device)
            for t in reversed(range(0, num_steps_denoise)):
                img = p_sample(model, img, torch.full((1,), t, device=device, dtype=torch.long), t)
            
            ready_batch_img = img.cpu().permute(0, 2, 3, 1).numpy()
               

            for b in range(batch):
                for n in range(n_cannels):
                    channel_path = os.path.join(out_dir, channel_names[n])
                    if not os.path.isdir(channel_path):
                        print("создаю out_dir:" + channel_path)
                        os.mkdir(channel_path)
                    
                        #io.imsave(os.path.join(out_dir, f"predict_{i}_step_denoise.png"), one_image , check_contrast=False)
                    convert_fun = img_to_uint if n == 0 else mask_to_uint
                    cv2.imwrite(os.path.join(channel_path, f"diffus_{img_index}.png"), convert_fun(ready_batch_img[b,:,:,n])) 
                img_index+=1
