# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室 
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#產生類比資料
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3 # y=2x，但是加入了噪聲
#圖形顯示
plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.legend()
plt.show()

#定義產生loss可視化的函數
plotdata = { "batchsize":[], "loss":[] }
def moving_average(a, w=10):
    if len(a) < w: 
        return a[:]    
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

tf.reset_default_graph()

# 定義研讀參數
training_epochs = 100  #設定迭代次數為58
display_step = 2

with tf.Session() as sess:
    savedir = "log2/"
    kpt = tf.train.latest_checkpoint(savedir)       #找到檢查點檔案
    print("kpt:",kpt)
    new_saver = tf.train.import_meta_graph(kpt+'.meta')  #從檢查點的meta檔案中匯入變數
    new_saver.restore(sess, kpt)                            #還原檢查點資料

    
    #裸取張量
    my_graph = tf.get_default_graph()
    #print(my_graph.get_operations())
    optimizer = my_graph.get_operation_by_name('GradientDescent')
    X = my_graph.get_tensor_by_name('Placeholder:0')
    Y = my_graph.get_tensor_by_name('Placeholder_1:0')
    cost = my_graph.get_tensor_by_name('Mean:0')
    result = my_graph.get_tensor_by_name('add:0')
    global_step = my_graph.get_tensor_by_name('global_step:0')
    


     
    # 向模型輸入資料
    while global_step.eval()/len(train_X) < training_epochs:
        step = int( global_step.eval()/len(train_X) )
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        #顯示訓練中的詳細訊息
        if step % display_step == 0:
            loss = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print ("Epoch:", step+1, "cost=", loss)
            if not (loss == "NA" ):
                plotdata["batchsize"].append(global_step.eval())
                plotdata["loss"].append(loss)
            new_saver.save(sess, savedir+"linermodel.cpkt", global_step)
                
    print (" Finished!")
    new_saver.save(sess, savedir+"linermodel.cpkt", global_step)
    
    print ("cost=", sess.run(cost, feed_dict={X: train_X, Y: train_Y}))


    
    plotdata["avgloss"] = moving_average(plotdata["loss"])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss')
     
    plt.show()
    
    graph_def = my_graph.as_graph_def()
    print(graph_def)
    tf.train.write_graph(graph_def, savedir, 'expert-graph.pb')
    #X = my_graph.get_tensor_by_name('Placeholder_2:0')
