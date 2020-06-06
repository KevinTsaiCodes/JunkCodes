# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室 
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""

import tensorflow as tf

#示範只有一個連續值特征列的動作
def test_one_column():                      
    price = tf.feature_column.numeric_column('price')          #定義一個特征列

    features = {'price': [[1.], [5.]]}          #將樣本資料定義為字典的型態
    net = tf.feature_column.input_layer(features, [price])     #將資料集與特征列一起輸入，到input_layer產生張量
    
    with tf.Session() as sess:                  #透過建立階段將其輸出
        tt  = sess.run(net)
        print( tt)

test_one_column()

#示範帶占位符的特征列動作
def test_placeholder_column():                      
    price = tf.feature_column.numeric_column('price')          #定義一個特征列

    features = {'price':tf.placeholder(dtype=tf.float64)}          #產生一個value為占位符的字典
    net = tf.feature_column.input_layer(features, [price])     #將資料集與特征列一起輸入，到input_layer產生張量
    
    with tf.Session() as sess:                  #透過建立階段將其輸出
        tt  = sess.run(net, feed_dict={
                features['price']: [[1.], [5.]]
            })
        print( tt)

test_placeholder_column()





import numpy as np
print(np.shape([[[1., 2.]], [[5., 6.]]]))
print(np.shape([[3., 4.], [7., 8.]]))
print(np.shape([[3., 4.]]))
def test_reshaping():
    tf.reset_default_graph()
    price = tf.feature_column.numeric_column('price', shape=[1, 2])#定義一個特征列,並指定形狀            
    features = {'price': [[[1., 2.]], [[5., 6.]]]}  #傳入一個3維的陣列
    features1 = {'price': [[3., 4.], [7., 8.]]}  #傳入一個2維的陣列

    
    net = tf.feature_column.input_layer(features, price)         #產生特征列張量
    net1 = tf.feature_column.input_layer(features1, price)         #產生特征列張量
    with tf.Session() as sess:                      #透過建立階段將其輸出
        print(net.eval())
        print(net1.eval())
        
test_reshaping()

def test_column_order():
    tf.reset_default_graph()
    price_a = tf.feature_column.numeric_column('price_a')   #定義了3個特征列 
    price_b = tf.feature_column.numeric_column('price_b')
    price_c = tf.feature_column.numeric_column('price_c')
    
    features = {                           #建立字典傳入資料
          'price_a': [[1.]],
          'price_c': [[4.]],          
          'price_b': [[3.]],
      }
    
    #產生輸入層
    net = tf.feature_column.input_layer(features, [price_c, price_a, price_b])
   
    with tf.Session() as sess:             #透過建立階段將其輸出
        print(net.eval())

test_column_order()        

