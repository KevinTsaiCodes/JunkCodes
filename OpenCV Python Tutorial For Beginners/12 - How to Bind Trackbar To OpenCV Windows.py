import cv2
import numpy as np

# Create a a black image, a window
img = np.zeros((300,512,3), np.uint8)
cv2.namedWindow('image')

def nothing(x):
    print(x)

cv2.createTrackbar('B','image',0,255,nothing)
cv2.createTrackbar('G','image',0,255,nothing)
cv2.createTrackbar('R','image',0,255,nothing)
# cv2.createTrackbar('trackBar_name','trackBar_in_Window_name',initial_value,final_value,change_values)

while True:
    cv2.imshow('image',img)
    if cv2.waitKey(1)&0xFF == ord('q'):
        break
    b = cv2.getTrackbarPos('B','image')
    g = cv2.getTrackbarPos('G','image')
    r = cv2.getTrackbarPos('R','image')
#   b = cv2.getTrackbarPos('traclBar_name','Window_name')
    img[:] = [b,g,r]
cv2.destroyAllWindows()
