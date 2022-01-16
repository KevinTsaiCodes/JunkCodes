import numpy as np
import matplotlib.pyplot as plt
import cv2

if __name__=='__main__':
    x = np.random.uniform(-1, 1, 4)
    y = np.random.uniform(-1, 1, 4)
    plt.plot(x,y,'r',label='line 1')
    plt.plot(x+2,y-20,'g--',label='line 2')
    plt.plot(x,20+y,'cyan',label='line 3')
    plt.plot(x+23,y-20,'b--',label='line 4')
    plt.scatter(x+10,y-5,label='dots')
    plt.legend()
    plt.show()
    
