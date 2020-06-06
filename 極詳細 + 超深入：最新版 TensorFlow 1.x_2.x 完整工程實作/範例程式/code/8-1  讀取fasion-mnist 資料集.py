# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室 
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./fashion/", one_hot=False)  #指定資料集路徑

print ('輸入資料:',mnist.train.images)
print ('輸入資料的形狀:',mnist.train.images.shape)
print ('輸入資料的標簽:',mnist.train.labels)

import pylab 
im = mnist.train.images[1]
im = im.reshape(-1,28)
pylab.imshow(im)
pylab.show()


#print ('輸入資料打shape:',mnist.test.images.shape)
#print ('輸入資料打shape:',mnist.validation.images.shape)
#print ('輸入資料:',mnist.test.labels)
















