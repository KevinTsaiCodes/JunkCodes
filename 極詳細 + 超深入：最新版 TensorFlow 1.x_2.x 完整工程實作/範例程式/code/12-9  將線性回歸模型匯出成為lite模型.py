# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室 
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""
#使用靜態圖訓練一個具有檢查點的回歸模型

import tensorflow as tf
#import numpy as np
#import matplotlib.pyplot as plt
#
##（1）產生類比資料
#train_X = np.linspace(-1, 1, 100)
#train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3 # y=2x，但是加入了噪聲
##圖形顯示
#plt.plot(train_X, train_Y, 'ro', label='Original data')
#plt.legend()
#plt.show()
#
#tf.reset_default_graph()
#
##（2）建立網路模型
#
## 建立模型
## 占位符
#X = tf.placeholder("float")
#Y = tf.placeholder("float")
## 模型參數
#W = tf.Variable(tf.random_normal([1]), name="weight")
#b = tf.Variable(tf.zeros([1]), name="bias")
## 前嚮結構
#z = tf.multiply(X, W)+ b
#global_step = tf.Variable(0, name='global_step', trainable=False)
##反向改善
#cost =tf.reduce_mean( tf.square(Y - z))
#learning_rate = 0.01
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,global_step) #梯度下降
#
## 起始化所有變數
#init = tf.global_variables_initializer()
## 定義研讀參數
#training_epochs = 28
#display_step = 2
#
#savedir = "log/"
#saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)#產生saver。 max_to_keep=1，表明最多只儲存一個檢查點檔案
#
##定義產生loss可視化的函數
#plotdata = { "batchsize":[], "loss":[] }
#def moving_average(a, w=10):
#    if len(a) < w: 
#        return a[:]    
#    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]
#
##（3）建立session進行訓練
#with tf.Session() as sess:
#    sess.run(init)
#    kpt = tf.train.latest_checkpoint(savedir)
#    if kpt!=None:
#        saver.restore(sess, kpt)
#     
#    # 向模型輸入資料
#    while global_step.eval()/len(train_X) < training_epochs:
#        step = int( global_step.eval()/len(train_X) )
#        for (x, y) in zip(train_X, train_Y):
#            sess.run(optimizer, feed_dict={X: x, Y: y})
#
#        #顯示訓練中的詳細訊息
#        if step % display_step == 0:
#            loss = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
#            print ("Epoch:", step+1, "cost=", loss,"W=", sess.run(W), "b=", sess.run(b))
#            if not (loss == "NA" ):
#                plotdata["batchsize"].append(global_step.eval())
#                plotdata["loss"].append(loss)
#            saver.save(sess, savedir+"linermodel.cpkt", global_step)
#                
#    print (" Finished!")
#    saver.save(sess, savedir+"linermodel.cpkt", global_step)
#    
#    print ("cost=", sess.run(cost, feed_dict={X: train_X, Y: train_Y}), "W=", sess.run(W), "b=", sess.run(b))
#
#    #顯示模型
#    plt.plot(train_X, train_Y, 'ro', label='Original data')
#    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
#    plt.legend()
#    plt.show()
#    
#    plotdata["avgloss"] = moving_average(plotdata["loss"])
#    plt.figure(1)
#    plt.subplot(211)
#    plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
#    plt.xlabel('Minibatch number')
#    plt.ylabel('Loss')
#    plt.title('Minibatch run vs. Training loss')
#     
#    plt.show()
#
#
#   
##################匯出模型#######################
#import os
#from tensorflow.python.tools import freeze_graph
#def exportmodel(thisgraph,saverex,thissavedir,outnode='',freeze_file_name  = 'expert-graph-yes.pb'):
#    
#    with tf.Session(graph=thisgraph) as sessex:
#    #sessex = tf.InteractiveSession()
#        sessex.run(tf.global_variables_initializer())
#        kpt = tf.train.latest_checkpoint(thissavedir)
#        
#        print("kpt:",kpt)
#    
#        if kpt!=None:
#            saverex.restore(sessex, kpt) 
#            
#            #print("xxxxx",x,result,saverex)
#            #graph_def = tf.get_default_graph().as_graph_def()
#            graph_def = thisgraph.as_graph_def()#取得全部的圖變數定義
#            tf.train.write_graph(graph_def, thissavedir, 'expert-graph.pb')#, as_text=False) 
#                
#            input_graph_path = os.path.join( thissavedir, 'expert-graph.pb')
#            input_saver_def_path = ""
#            input_binary = False
#            #匯出的名字，後面不能用：0，  但是載入時需要Placeholder_2:0
#            output_node_names = outnode  #也可以不寫
#            restore_op_name = "save/restore_all"
#            filename_tensor_name = "save/Const:0"
#            output_graph_path = os.path.join(thissavedir, freeze_file_name)
#            clear_devices = True
#            input_meta_graph = ""
#            
#            freeze_graph.freeze_graph(
#                    input_graph_path, input_saver_def_path, input_binary, kpt,
#                    output_node_names, restore_op_name, filename_tensor_name,
#                    output_graph_path, clear_devices, "", "") 
#                    
#print(z.name,X.name)#將節點列印出來
#                    
#thisgraph = tf.get_default_graph()
#saverex = tf.train.Saver() # 產生saver
#exportmodel(thisgraph,saverex,savedir,"add,Placeholder")       

import tensorflow as tf
        
tf.reset_default_graph()
#savedir = "log/"
#graph_def_file = savedir+"expert-graph-yes.pb"
#print(graph_def_file)
#input_arrays = ["Placeholder"]
#output_arrays = ["add"] 
#       
#converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(
##converter = tf.lite.TFLiteConverter.from_frozen_graph(
#  graph_def_file, input_arrays, output_arrays)
#tflite_model = converter.convert()
#open("converted_model.tflite", "wb").write(tflite_model)

import tensorflow as tf

img = tf.placeholder(name="img", dtype=tf.float32, shape=(1, 64, 64, 3))
var = tf.get_variable("weights", dtype=tf.float32, shape=(1, 64, 64, 3))
val = img + var
out = tf.identity(val, name="out")

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  converter = tf.contrib.lite.TFLiteConverter.from_session(sess, [img], [out])
  tflite_model = converter.convert()
  open("converted_model.tflite", "wb").write(tflite_model)


       