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
model_path =  "epoch_46_model_by_config_test_&_tiny_unet_v3_!.pt"
#model_path =  "model_by_config_test_&_tiny_unet_v3_!.pt"

if os.path.isfile(model_path):
    print("File is found")
else:
    print("ERROR file founded")

model = torch.load(model_path)
last_activation="sigmoid_activation"


EPSILON = 1e-7


num_steps_denoise = 800

device = "cuda"
out_dir = "result"

#def predictModel(model, device, last_activation, eps = EPSILON):
result = []
model.to(device)
model.eval()


betas = torch.linspace(0.0001, 0.02, num_steps_denoise)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


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
    output=model(x)
    res = output
    #res = globals()[last_activation](output, EPSILON)
    
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

with torch.no_grad():
    result=[]
    img = torch.randn((1,1, 256,256), device=device)
    result.append(img.cpu().permute(0, 2, 3, 1).numpy())
    for t in tqdm.tqdm(reversed(range(0, num_steps_denoise)), desc='sampling loop time step', total=num_steps_denoise):
        img = p_sample(model, img, torch.full((1,), t, device=device, dtype=torch.long), t)
        result.append(img.cpu().permute(0, 2, 3, 1).numpy())
        




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

import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()
ims = []
for i, image in enumerate(result):
    if i%2==0:
        im = plt.imshow(image[0,:,:,0], cmap="gray", animated=True)
        ims.append([im])

animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=5000)
animate.save('diffusion.gif')
