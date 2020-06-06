# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""

import tensorflow as tf




dataset1 = tf.data.Dataset.from_tensor_slices( [1,2,3,4,5] )	#定義訓練資料集

#建立迭代器
iterator1 = tf.data.Iterator.from_structure(dataset1.output_types,dataset1.output_shapes)

one_element1 = iterator1.get_next()

with tf.Session()  as sess2:
    sess2.run( iterator1.make_initializer(dataset1) )#起始化迭代器
    for ii in range(2):  #資料集迭代兩次
        while True:		#透過for循環列印所有的資料
            try:
                print(sess2.run(one_element1))				#呼叫sess.run讀出Tensor值
            except tf.errors.OutOfRangeError:
                print("檢查結束")
                sess2.run( iterator1.make_initializer(dataset1) )# 從頭再來一遍
                break


    print(sess2.run(one_element1,{one_element1:356}))  #往資料集中植入資料


dataset1 = tf.data.Dataset.from_tensor_slices( [1,2,3,4,5] )	#定義訓練資料集
iterator = dataset1.make_one_shot_iterator()  #產生一個迭代器

dataset_test = tf.data.Dataset.from_tensor_slices( [10,20,30,40,50] )#定義測試資料集
iterator_test = dataset1.make_one_shot_iterator()  #產生一個迭代器
#適用於測試與訓練場景下的資料集模式
with tf.Session()  as sess:
    iterator_handle = sess.run(iterator.string_handle())
    iterator_handle_test = sess.run(iterator_test.string_handle())

    handle = tf.placeholder(tf.string, shape=[])
    iterator3 = tf.data.Iterator.from_string_handle(handle, iterator.output_types)

    one_element3 = iterator3.get_next()
    print(sess.run(one_element3,{handle: iterator_handle}))
    print(sess.run(one_element3,{handle: iterator_handle_test}))














