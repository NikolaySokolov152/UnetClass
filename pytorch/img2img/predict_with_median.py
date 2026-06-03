import cv2
import os
import torch
import numpy as np
import skimage.io as io

import sys
if __name__ == "__main__":
   sys.path.append("../src/")
   sys.path.append("../")

model_path = "2024_04_14/model_by_config_tiny_unet_v3.pt"
img_dir = "F:/репозитории и код/Synthetics/dataset/synthetic_dataset10/original"

eps = 0
@torch.jit.script
def sigmoid_activation(x : torch.Tensor, epsilon : float) -> torch.Tensor:
    return torch.sigmoid(x)

model = torch.load(model_path)
img_type = ('.png', '.jpg', '.jpeg')
img_names = [name for name in os.listdir(img_dir) if name.endswith((img_type))]


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)


for img_name in img_names:
    #img_name = img_name
    #print(img_dir)    
    #print(img_name)
    img_path = os.path.join(img_dir, img_name)
    img = io.imread(img_path, as_gray=True)
    #img = np.expand_dims(img, axis=-1)
    if img is None:
        raise Exception(f"can't open image '{img_name}'")

    if img.dtype.kind == 'f':
        print(f"type image '{img_name}' open as float")
        if img.max() <= 1:
            img = img*255
        img = img.astype(np.uint8)
        


    #cv2.imshow(f"original", img)
    for k in range(1, 2):
        k = 4
        d = 2 * k + 1
        median = cv2.medianBlur(img, d)
        #median = img

        median = median.astype(float)
        median/=255
        item = np.reshape(median, (1,) + median.shape + (1,))
        torch_img = torch.from_numpy(np.array(item)).type(torch.FloatTensor).permute(0, 3, 1, 2).to(device)
        inputs = torch_img.to(device)
        outputs = model(inputs)

        outputs = sigmoid_activation(outputs, eps)

        result = outputs.detach().cpu().permute(0, 2, 3, 1).numpy()[0]

        cv2.imshow(f"median kernel {k}", result.copy())
        #cv2.imwrite(f"result_no_median/{img_name}.png", (result*255).astype(np.uint8))
    #cv2.waitKey()
