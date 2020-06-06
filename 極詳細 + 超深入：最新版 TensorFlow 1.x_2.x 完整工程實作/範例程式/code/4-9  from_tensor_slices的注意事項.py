# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""

import tensorflow as tf



dataset1 = tf.data.Dataset.from_tensor_slices( [1,2,3,4,5] )
#dataset1 = tf.data.Dataset.from_tensor_slices( (1,2,3,4,5) )
#dataset1 = tf.data.Dataset.from_tensor_slices( ([1],[2],[3],[4],[5]) )

def getone(dataset):
    iterator = dataset.make_one_shot_iterator()			#產生一個迭代器
    one_element = iterator.get_next()					#從iterator裡取出一個元素
    return one_element

one_element1 = getone(dataset1)

with tf.Session() as sess:	# 建立階段（session）
    for i in range(5):		#透過for循環列印所有的資料
        print(sess.run(one_element1))				#呼叫sess.run讀出Tensor值




dataset1 = tf.data.Dataset.from_tensor_slices( ([1],[2],[3],[4],[5]) )
one_element1 = getone(dataset1)
with tf.Session() as sess:	# 建立階段（session）
    print(sess.run(one_element1))				#呼叫sess.run讀出Tensor值