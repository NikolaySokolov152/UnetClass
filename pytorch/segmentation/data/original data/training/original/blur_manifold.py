import cv2




img = cv2.imread("training0000.png", 0)


for k in range(1, 5):

    d = 2*k+1
    
    bilat = cv2.bilateralFilter(img, d, 10, 5)
    
    cv2.imshow(f"bilateralFilter diam {d}", bilat)
    
cv2.waitKey()