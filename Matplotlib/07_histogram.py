import numpy as np
import matplotlib.pyplot as plt
import cv2

if __name__=='__main__':
    img = cv2.imread('data/sign.jpg', cv2.IMREAD_GRAYSCALE)
    _, thresh_img = cv2.threshold(img, 70, 255, cv2.THRESH_TOZERO)
    cv2.imshow('threshold', thresh_img)
    cv2.waitKey(0)
    plt.hist(thresh_img.ravel(), bins=256, range=(0.0, 255.0))
    plt.show()
