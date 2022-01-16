import numpy as np
import matplotlib.pyplot as plt


if __name__=='__main__':
    data = np.arange(0., 5., 0.2)
    plt.scatter(data,data, data,data**2,'g^', data/2,data**2,'b-.', data/3,data*np.log(2)+2,'cyan')
    # above shows how to plot multiplot in one plot graph
    # as following, we use plt.plot(dataX1,dataY1,style1, dataX2,dataY2, style2, dataXN,dataYN,styleN)
    plt.show()
