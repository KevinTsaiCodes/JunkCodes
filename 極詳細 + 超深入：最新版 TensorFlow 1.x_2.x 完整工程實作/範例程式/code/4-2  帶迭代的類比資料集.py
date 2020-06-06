# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室 
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
#在記憶體中產生類比資料
def GenerateData(training_epochs ,batchsize = 100):
    for i in range(training_epochs):
        train_X = np.linspace(-1, 1, batchsize)   #train_X為-1到1之間連續的100個浮點數
        train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3 # y=2x，但是加入了噪聲
        
        yield shuffle(train_X, train_Y),i

Xinput = tf.placeholder("float",(None))  #定義兩個占位符，用來接收參數
Yinput = tf.placeholder("float",(None))


training_epochs = 20  # 定義需要迭代的次數

with tf.Session() as sess:  # 建立階段（session）

    for (x, y) ,ii in GenerateData(training_epochs):
        xv,yv = sess.run([Xinput,Yinput],feed_dict={Xinput: x, Yinput: y})#透過靜態圖植入的模式，傳入資料
        print(ii,"| x.shape:",np.shape(xv),"| x[:3]:",xv[:3])
        print(ii,"| y.shape:",np.shape(yv),"| y[:3]:",yv[:3])

     
    
#顯示類比資料點
train_data =list(GenerateData(1))[0]
plt.plot(train_data[0][0], train_data[0][1], 'ro', label='Original data')
plt.legend()
plt.show()