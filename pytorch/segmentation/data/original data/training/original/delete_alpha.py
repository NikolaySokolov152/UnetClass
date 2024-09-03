import cv2
import os 


images = [name for name in os.listdir("./") if name.endswith(".png")]

for name in images:
    img = cv2.imread(name)
    if img.shape[2] %2 == 0:   
        img = img[...,:img.shape[2]-1]
    
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(name, img)