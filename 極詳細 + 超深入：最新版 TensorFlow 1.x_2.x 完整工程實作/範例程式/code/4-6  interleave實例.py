# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""



import tensorflow as tf
Dataset =tf.data.Dataset

def parse_fn(x):
    print(x)
    return x

dataset = (Dataset.list_files('testset\*.txt', shuffle=False)
               .interleave(lambda x:
                   tf.data.TextLineDataset(x).map(parse_fn, num_parallel_calls=1),
                   cycle_length=2, block_length=2))




def getone(dataset):
    iterator = dataset.make_one_shot_iterator()			#產生一個迭代器
    one_element = iterator.get_next()					#從iterator裡取出一個元素
    return one_element

one_element1 = getone(dataset)				#從dataset裡取出一個元素


def showone(one_element,datasetname):
    print('{0:-^50}'.format(datasetname))
    for ii in range(20):
        datav = sess.run(one_element)#透過靜態圖植入的模式，傳入資料
        print(datav)



with tf.Session() as sess:	# 建立階段（session）
    showone(one_element1,"dataset1")
