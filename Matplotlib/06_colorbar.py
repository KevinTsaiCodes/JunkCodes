import numpy as np
import matplotlib.pyplot as plt
import cv2

if __name__=='__main__':
    img = cv2.imread('data/lena.jpg')
    plt.imshow(img)
    plt.colorbar()
    plt.show()
