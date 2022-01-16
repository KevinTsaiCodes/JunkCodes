import numpy as np
import matplotlib.pyplot as plt


if __name__=='__main__':
    data = np.arange(0,100,3)
    plt.scatter(data, 100-data*np.cos(30), 12, 'r')
