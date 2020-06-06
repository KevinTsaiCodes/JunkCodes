# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室 
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""


import tensorflow as tf
import numpy as np

#在記憶體中產生類比資料
def GenerateData(datasize = 100 ):
    train_X = np.linspace(-1, 1, datasize)   #train_X為-1到1之間連續的100個浮點數
    train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3 # y=2x，但是加入了噪聲
    return train_X, train_Y   #以產生器的模式傳回


train_data = GenerateData()  

#將記憶體資料轉化成資料集
dataset = tf.data.Dataset.from_tensor_slices( train_data )		#元祖
dataset2 = tf.data.Dataset.from_tensor_slices( {                #字典
        "x":train_data[0],
        "y":train_data[1]
        } )		#

batchsize = 10  #定義批次樣本個數
dataset3 = dataset.repeat().batch(batchsize) #批次劃分資料集

dataset4 = dataset2.map(lambda data: (data['x'],tf.cast(data['y'],tf.int32)) )#自訂處理資料集元素
dataset5 = dataset.shuffle(100)#亂序資料集
    
def getone(dataset):
    iterator = dataset.make_one_shot_iterator()			#產生一個迭代器
    one_element = iterator.get_next()					#從iterator裡取出一個元素  
    return one_element
    
one_element1 = getone(dataset)				#從dataset裡取出一個元素
one_element2 = getone(dataset2)				#從dataset2裡取出一個元素
one_element3 = getone(dataset3)				#從dataset3裡取出一個批次的元素
one_element4 = getone(dataset4)				#從dataset4裡取出一個批次的元素
one_element5 = getone(dataset5)				#從dataset5裡取出一個批次的元素


def showone(one_element,datasetname):
    print('{0:-^50}'.format(datasetname))
    for ii in range(5):
        datav = sess.run(one_element)#透過靜態圖植入的模式，傳入資料
        print(datasetname,"-",ii,"| x,y:",datav)
        
def showbatch(onebatch_element,datasetname):
    print('{0:-^50}'.format(datasetname))
    for ii in range(5):
        datav = sess.run(onebatch_element)#透過靜態圖植入的模式，傳入資料
        print(datasetname,"-",ii,"| x.shape:",np.shape(datav[0]),"| x[:3]:",datav[0][:3])
        print(datasetname,"-",ii,"| y.shape:",np.shape(datav[1]),"| y[:3]:",datav[1][:3])
        
with tf.Session() as sess:	# 建立階段（session）
    showone(one_element1,"dataset1")
    showone(one_element2,"dataset2")
    showbatch(one_element3,"dataset3")
    showone(one_element4,"dataset4")
    showone(one_element5,"dataset5")
    

    

