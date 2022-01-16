import numpy as np
import matplotlib.pyplot as plt
import cv2

if __name__=='__main__':
    img = cv2.imread('data/lena.jpg')
    img = img[:,:,0]
    plt.imshow(img)
