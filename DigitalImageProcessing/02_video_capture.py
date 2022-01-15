import cv2


if __name__=='__main__':
    cap = cv2.VideoCapture(0) # if not work with your camera, start with -1
    print(cap.isOpened()) # returns true if video capturing has been initialized already.
    while True:
        _, frame = cap.read()
        # read([, image]) -> retval, image.   
        # @brief Grabs, decodes and returns the next video frame.
        
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(1)&0xFF == ord('q'):
            break
        
    cap.release() # Closes video file or capturing device.
    cv2.destroyAllWindows()
