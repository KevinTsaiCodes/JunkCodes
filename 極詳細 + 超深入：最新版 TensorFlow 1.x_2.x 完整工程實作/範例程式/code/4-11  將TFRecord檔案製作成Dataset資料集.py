# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室 
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""

import tensorflow as tf 
from PIL import Image 
import numpy as np
import matplotlib.pyplot as plt

def dataset(directory,size,batchsize):#定義函數，建立資料集
    """ parse  dataset."""
    def _parseone(example_proto):                         #解析一個圖片檔案
        """ Reading and handle  image"""
        #定義解析的字典
        dics = {}
        dics['label'] = tf.FixedLenFeature(shape=[],dtype=tf.int64)
        dics['img_raw'] = tf.FixedLenFeature(shape=[],dtype=tf.string)

        parsed_example = tf.parse_single_example(example_proto,dics)#解析一行樣本
        
        image = tf.decode_raw(parsed_example['img_raw'],out_type=tf.uint8)
        image = tf.reshape(image, size)
        image = tf.cast(image,tf.float32)*(1./255)-0.5 #對圖形資料做歸一化
        
        label = parsed_example['label']
        label = tf.cast(label,tf.int32)
        label = tf.one_hot(label, depth=2, on_value=1) #轉為0ne-hot解碼
        return image,label
    
    dataset = tf.data.TFRecordDataset(directory)
    dataset = dataset.map(_parseone)
    dataset = dataset.batch(batchsize) #批次劃分資料集
    
    dataset = dataset.prefetch(batchsize)
                
    return dataset


#若果顯示有錯，可以嘗試使用np.reshape(thisimg, (size[0],size[1],3))或
#np.asarray(thisimg[0], dtype='uint8')改變型態與形狀
def showresult(subplot,title,thisimg):          #顯示單一圖片
    p =plt.subplot(subplot)
    p.axis('off') 
    p.imshow(thisimg)
    p.set_title(title)

def showimg(index,label,img,ntop):   #顯示 
    plt.figure(figsize=(20,10))     #定義顯示圖片的寬、高
    plt.axis('off')  
    ntop = min(ntop,9)
    print(index)
    for i in range (ntop):
        showresult(100+10*ntop+1+i,label[i],img[i])  
    plt.show() 

def getone(dataset):
    iterator = dataset.make_one_shot_iterator()			#產生一個迭代器
    one_element = iterator.get_next()					#從iterator裡取出一個元素  
    return one_element

sample_dir=['mydata.tfrecords']
size = [256,256,3]
batchsize = 10
tdataset = dataset(sample_dir,size,batchsize)

print(tdataset.output_types)  #列印資料集的輸出訊息
print(tdataset.output_shapes)

one_element1 = getone(tdataset)				#從tdataset裡取出一個元素

with tf.Session() as sess:	# 建立階段（session）
    sess.run(tf.global_variables_initializer())  #起始化
    try:
        for step in np.arange(1):
            value = sess.run(one_element1)
            showimg(step,value[1],np.asarray( (value[0]+0.5)*255,np.uint8),10)       #顯示圖片        
    except tf.errors.OutOfRangeError:           #捕捉例外
        print("Done!!!")
