# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室 
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""
import tensorflow as tf
from tensorflow.python.feature_column.feature_column import _LazyBuilder

#將離散文字按照指定範圍雜湊
def test_categorical_cols_to_hash_bucket():
    tf.reset_default_graph()
    some_sparse_column = tf.feature_column.categorical_column_with_hash_bucket(
        'sparse_feature', hash_bucket_size=5)#稀疏矩陣，單獨放進去會出錯
   
    builder = _LazyBuilder({
          'sparse_feature': [['a'], ['x']],
      })
    id_weight_pair = some_sparse_column._get_sparse_tensors(builder) #

    with tf.Session() as sess:
        
        id_tensor_eval = id_weight_pair.id_tensor.eval()
        print("稀疏矩陣：\n",id_tensor_eval)
          
        dense_decoded = tf.sparse_tensor_to_dense( id_tensor_eval, default_value=-1).eval(session=sess)
        print("稠密矩陣：\n",dense_decoded)

test_categorical_cols_to_hash_bucket()


from tensorflow.python.ops import lookup_ops
#將離散文字按照指定詞表與指定範圍，混合雜湊
def test_with_1d_sparse_tensor():
    tf.reset_default_graph()
    #混合雜湊
    body_style = tf.feature_column.categorical_column_with_vocabulary_list(
        'name', vocabulary_list=['anna', 'gary', 'bob'],num_oov_buckets=2)   #稀疏矩陣
    
    #稠密矩陣
    builder = _LazyBuilder({
        'name': ['anna', 'gary','alsa'],        
      })
    
    #稀疏矩陣
    builder2 = _LazyBuilder({
        'name': tf.SparseTensor(
        indices=((0,), (1,), (2,)),
        values=('anna', 'gary', 'alsa'),
        dense_shape=(3,)),    
      })    

    id_weight_pair = body_style._get_sparse_tensors(builder)    #
    id_weight_pair2 = body_style._get_sparse_tensors(builder2)  #


    with tf.Session() as sess:
        sess.run(lookup_ops.tables_initializer())

        id_tensor_eval = id_weight_pair.id_tensor.eval()
        print("稀疏矩陣：\n",id_tensor_eval)
        id_tensor_eval2 = id_weight_pair2.id_tensor.eval()
        print("稀疏矩陣2：\n",id_tensor_eval2)
          
        dense_decoded = tf.sparse_tensor_to_dense( id_tensor_eval, default_value=-1).eval(session=sess)
        print("稠密矩陣：\n",dense_decoded)

test_with_1d_sparse_tensor()


#將離散文字轉為onehot特征列
def test_categorical_cols_to_onehot():
    tf.reset_default_graph()
    some_sparse_column = tf.feature_column.categorical_column_with_hash_bucket(
        'sparse_feature', hash_bucket_size=5)                                       #定義雜湊特征列
    
    #轉換成one-hot特征列
    one_hot_style = tf.feature_column.indicator_column(some_sparse_column)     
  

    features = {
      'sparse_feature': [['a'], ['x']],
      }

    net = tf.feature_column.input_layer(features, one_hot_style)               #產生輸入層張量
    with tf.Session() as sess:                                                      #透過階段輸出資料
        print(net.eval()) 

test_categorical_cols_to_onehot()





#將離散文字轉為onehot詞內嵌特征列
def test_categorical_cols_to_embedding():
    tf.reset_default_graph()
    some_sparse_column = tf.feature_column.categorical_column_with_hash_bucket(
        'sparse_feature', hash_bucket_size=5)#稀疏矩陣，單獨放進去會出錯
   
    embedding_col = tf.feature_column.embedding_column( some_sparse_column, dimension=3)

    features = {
          'sparse_feature': [['a'], ['x']],
      }
    
    #產生輸入層張量
    cols_to_vars = {}
    net = tf.feature_column.input_layer(features, embedding_col,cols_to_vars)
  
    with tf.Session() as sess:                  #透過階段輸出資料
        sess.run(tf.global_variables_initializer())
        print(net.eval()) 

test_categorical_cols_to_embedding()

#input_layer中的順序
def test_order():
    tf.reset_default_graph()
    numeric_col = tf.feature_column.numeric_column('numeric_col')
    some_sparse_column = tf.feature_column.categorical_column_with_hash_bucket(
        'asparse_feature', hash_bucket_size=5)#稀疏矩陣，單獨放進去會出錯
   
    embedding_col = tf.feature_column.embedding_column( some_sparse_column, dimension=3)
    #轉換成one-hot特征列
    one_hot_col = tf.feature_column.indicator_column(some_sparse_column)
    print(one_hot_col.name)
    print(embedding_col.name)
    print(numeric_col.name)

    features = {
          'numeric_col': [[3], [6]],
          'asparse_feature': [['a'], ['x']],
      }
    
    #產生輸入層張量
    cols_to_vars = {}
    net = tf.feature_column.input_layer(features, [numeric_col,one_hot_col,embedding_col],cols_to_vars)

    with tf.Session() as sess:                  #透過階段輸出資料
        sess.run(tf.global_variables_initializer())
        print(net.eval()) 

test_order()
