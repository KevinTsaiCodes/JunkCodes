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

#在記憶體中產生類比資料
def GenerateData(batchsize = 100):
    train_X = np.linspace(-1, 1, batchsize)   #train_X為-1到1之間連續的100個浮點數
    train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3 # y=2x，但是加入了噪聲
    yield train_X, train_Y       #以產生器的模式傳回

#定義網路模型結構部分，這裡只有占位符張量
Xinput = tf.placeholder("float",(None))  #定義兩個占位符，用來接收參數
Yinput = tf.placeholder("float",(None))

#建立階段，取得並輸出資料
training_epochs = 20  # 定義需要迭代的次數
with tf.Session() as sess:  # 建立階段（session）
    for epoch in range(training_epochs): #迭代資料集20遍
        for x, y in GenerateData(): #透過for循環列印所有的點
            xv,yv = sess.run([Xinput,Yinput],feed_dict={Xinput: x, Yinput: y})#透過靜態圖植入的模式，傳入資料

            print(epoch,"| x.shape:",np.shape(xv),"| x[:3]:",xv[:3])
            print(epoch,"| y.shape:",np.shape(yv),"| y[:3]:",yv[:3])
     
    
#顯示類比資料點
train_data =list(GenerateData())[0]
plt.plot(train_data[0], train_data[1], 'ro', label='Original data')
plt.legend()
plt.show()