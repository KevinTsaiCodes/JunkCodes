# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室 
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""

import tensorflow as tf
from tensorflow.python.feature_column.feature_column import _LazyBuilder

def test_crossed():   
    
    a = tf.feature_column.numeric_column('a', dtype=tf.int32, shape=(2,))
    b = tf.feature_column.bucketized_column(a, boundaries=(0, 1))               #離散值轉化    
    crossed = tf.feature_column.crossed_column([b, 'c'], hash_bucket_size=5)#產生交叉列

    builder = _LazyBuilder({                                                #產生類比輸入的資料
          'a':
              tf.constant(((-1.,-1.5), (.5, 1.))),
          'c':
              tf.SparseTensor(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=['cA', 'cB', 'cC'],
                  dense_shape=(2, 2)),
      })
    id_weight_pair = crossed._get_sparse_tensors(builder)#產生輸入層張量      
    with tf.Session() as sess2:                             #建立階段session，取值
          id_tensor_eval = id_weight_pair.id_tensor.eval()
          print(id_tensor_eval)                             #輸出稀疏矩陣
          
          dense_decoded = tf.sparse_tensor_to_dense( id_tensor_eval, default_value=-1).eval(session=sess2)
          print(dense_decoded)                               #輸出稠密矩陣
          
test_crossed()

