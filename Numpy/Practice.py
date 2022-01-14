import numpy as np

if __name__=='__main__':
    df = np.genfromtxt('data.txt', delimiter=',')
    x = df[0,:]
    y = df[1,:]
    x = np.reshape(x, (2,3))
    y = np.reshape(y, (3,2))
    z = np.matmul(x,y)
    print(x)
    print(y)
    print(z)
