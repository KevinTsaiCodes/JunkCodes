import cv2
import PIL
from PIL import Image
import numpy as np

if __name__ == '__main__':
    img = cv2.imread('data/lena.jpg', cv2.IMREAD_GRAYSCALE)
    print(img) # return None, means no image
    im_pil = Image.fromarray(img) # change opencv data to pillow data
    im_pil.show()
    # For reversing the operation:
    im_np = np.asarray(im_pil)
    print(im_np)
    cv2.imshow('lena', img) # opencv show image
    cv2.waitKey(0) # wait some seconds, in order to show opencv data
    cv2.destroyAllWindows() # destroy All opencv data windows
    # cv2.destroyWindow('lena') # destroy specific opencv data window
    cv2.imwrite('data/lena_cpy_opencv.png', img) # save the opencv data file
    im_pil.save('data/lena_cpy_pillow.png') # save the pillow data file
