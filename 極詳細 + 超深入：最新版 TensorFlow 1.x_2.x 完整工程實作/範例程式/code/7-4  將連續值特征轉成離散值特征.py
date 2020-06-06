# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室 
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""
import tensorflow as tf

def test_numeric_cols_to_bucketized():
    price = tf.feature_column.numeric_column('price')#定義連續數值的特征列

    #將連續數值轉成離散值的特征列,離散值共分為3段：小於3、在3與5之間、大於5
    price_bucketized = tf.feature_column.bucketized_column(  price, boundaries=[3.,5.])

    features = {                        #傳定義字典
          'price': [[2.], [6.]],
      }

    net = tf.feature_column.input_layer(features,[ price,price_bucketized]) #產生輸入層張量
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(net.eval()) 

test_numeric_cols_to_bucketized()

def test_numeric_cols_to_identity():
    tf.reset_default_graph()
    price = tf.feature_column.numeric_column('price')#定義連續數值的特征列

    categorical_column = tf.feature_column.categorical_column_with_identity('price', 6)
    print(type(categorical_column))
    one_hot_style = tf.feature_column.indicator_column(categorical_column)
    features = {                        #傳定義字典
          'price': [[2], [4]],
      }

    net = tf.feature_column.input_layer(features,[ price,one_hot_style]) #產生輸入層張量
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(net.eval()) 

test_numeric_cols_to_identity()