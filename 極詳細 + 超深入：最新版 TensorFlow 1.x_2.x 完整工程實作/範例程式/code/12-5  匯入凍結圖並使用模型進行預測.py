# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室 
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""
import tensorflow as tf
        
tf.reset_default_graph()
 
savedir = "log/"
PATH_TO_CKPT = savedir +'/expert-graph-yes.pb'

my_graph_def = tf.GraphDef() #定義GraphDef物件
with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    my_graph_def.ParseFromString(serialized_graph)#讀pb檔案
    print(my_graph_def)
    tf.import_graph_def(my_graph_def, name='')#還原到目前圖中

my_graph = tf.get_default_graph()  #獲得目前圖
result = my_graph.get_tensor_by_name('add:0')#獲得目前圖中的z給予值給result
x = my_graph.get_tensor_by_name('Placeholder:0')#獲得目前圖中的X給予值給x    
        
with tf.Session() as sess:
    y = sess.run(result, feed_dict={x: 5})#傳入5，進行預測
    print(y)
        

       