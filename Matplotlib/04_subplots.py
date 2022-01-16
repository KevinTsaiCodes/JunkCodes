import numpy as np
import matplotlib.pyplot as plt


if __name__=='__main__':
    data = np.arange(0,100,3)
    plt.subplot(2,3,1)
    plt.bar(data, data)
    plt.subplot(2,3,4)
    plt.scatter(data, data)
    plt.subplot(2,3,3)
    plt.plot(data, data)
    plt.suptitle('Categorical Plotting')
    plt.show()
    # Last you need to add plt.show() in order to show the plot
    
    # creating subplots, using subplot(X_length,Y_length,position)
    # X_lenght*Y_length = Full Plot Area
    # Example:  subplot(2,4); means the full plot is 2*4
    # Position: 1 2 3 4
    #           5 6 7 8
