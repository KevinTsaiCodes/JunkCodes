# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室 
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""

import tensorflow as tf

tf.reset_default_graph()

vocabulary_size = 3                           #假如有3個詞，向量為0，1，2

sparse_input_a = tf.SparseTensor(             #定義一個稀疏矩陣,  值為：  
    indices=((0, 0), (1, 0), (1, 1)),         #      [2]   只有一個序列
    values=(2, 0, 1),                         #      [0, 1] 有連個序列
    dense_shape=(2, 2))

sparse_input_b = tf.SparseTensor(             #定義一個稀疏矩陣,  值為：  
    indices=((0, 0), (1, 0), (1, 1)),         #      [1]
    values=(1, 2, 0),                         #      [2, 0]
    dense_shape=(2, 2))

embedding_dimension_a = 2
embedding_values_a = (                      #為稀疏矩陣的三個值（0，1，2）比對詞內嵌初值
    (1., 2.),  # id 0
    (3., 4.),  # id 1
    (5., 6.)  # id 2
)
embedding_dimension_b = 3
embedding_values_b = (                     #為稀疏矩陣的三個值（0，1，2）比對詞內嵌初值
    (11., 12., 13.),  # id 0
    (14., 15., 16.),  # id 1
    (17., 18., 19.)  # id 2
)

def _get_initializer(embedding_dimension, embedding_values): #自訂起始化詞內嵌
  def _initializer(shape, dtype, partition_info):
    return embedding_values
  return _initializer



categorical_column_a = tf.contrib.feature_column.sequence_categorical_column_with_identity( #帶序列的離雜湊
    key='a', num_buckets=vocabulary_size)
embedding_column_a = tf.feature_column.embedding_column(    #將離雜湊轉為詞向量
    categorical_column_a, dimension=embedding_dimension_a,
    initializer=_get_initializer(embedding_dimension_a, embedding_values_a))


categorical_column_b = tf.contrib.feature_column.sequence_categorical_column_with_identity(
    key='b', num_buckets=vocabulary_size)
embedding_column_b = tf.feature_column.embedding_column(
    categorical_column_b, dimension=embedding_dimension_b,
    initializer=_get_initializer(embedding_dimension_b, embedding_values_b))



shared_embedding_columns = tf.feature_column.shared_embedding_columns( #共享列
        [categorical_column_b, categorical_column_a],
        dimension=embedding_dimension_a,
        initializer=_get_initializer(embedding_dimension_a, embedding_values_a))
 
features={                                              #將a,b合起來
        'a': sparse_input_a,
        'b': sparse_input_b,
    }

input_layer, sequence_length = tf.contrib.feature_column.sequence_input_layer(    #定義序列輸入層
        features,
    feature_columns=[embedding_column_b, embedding_column_a])

input_layer2, sequence_length2 = tf.contrib.feature_column.sequence_input_layer(    #定義序列輸入層
        features,
    feature_columns=shared_embedding_columns)

global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)#傳回圖中的張量（2個內嵌詞權重）
print([v.name for v in global_vars])  

with tf.train.MonitoredSession() as sess:
    print(global_vars[0].eval(session=sess))#輸出詞向量的初值
    print(global_vars[1].eval(session=sess))
    print(global_vars[2].eval(session=sess))
    print(sequence_length.eval(session=sess))
    print(input_layer.eval(session=sess))   #輸出序列輸入層的內容
    print(sequence_length2.eval(session=sess))  
    print(input_layer2.eval(session=sess))   #輸出序列輸入層的內容

    
