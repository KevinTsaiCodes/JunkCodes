import cv2
import matplotlib.pyplot as plt

path = "C:/Users/twc04/OneDrive/Desktop/opencv/samples/data/lena.jpg"

img = cv2.imread(path)

plt.imshow(img) # use matplotlib
cv2.imshow('img',img) # use opencv

cv2.waitKey(0)
cv2.destroyAllWindows()
