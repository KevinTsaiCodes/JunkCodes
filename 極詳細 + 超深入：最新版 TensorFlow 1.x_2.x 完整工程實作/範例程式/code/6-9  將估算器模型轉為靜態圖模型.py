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

#在記憶體中產生類比資料
def GenerateData(datasize = 100 ):
    train_X = np.linspace(-1, 1, datasize)   #train_X為-1到1之間連續的100個浮點數
    train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3 # y=2x，但是加入了噪聲
    return train_X, train_Y   #以產生器的模式傳回

train_data = GenerateData()  

batch_size=10

def train_input_fn(train_data, batch_size):  #定義訓練資料集輸入函數
    #建構資料集的群組成：一個特征輸入，一個標簽輸入
    dataset = tf.data.Dataset.from_tensor_slices( (  train_data[0],train_data[1]) )   
    dataset = dataset.shuffle(1000).repeat().batch(batch_size) #將資料集亂序、重復、批次劃分. 
    return dataset     #傳回資料集 



#定義產生loss可視化的函數
plotdata = { "batchsize":[], "loss":[] }
def moving_average(a, w=10):
    if len(a) < w: 
        return a[:]    
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

tf.reset_default_graph()


features = tf.placeholder("float",[None])#重新定義占位符
labels = tf.placeholder("float",[None])

#其他網路結構不變
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")
predictions = tf.multiply(tf.cast(features,dtype = tf.float32), W)+ b# 前嚮結構
loss = tf.losses.mean_squared_error(labels=labels, predictions=predictions)#定義損失函數

global_step = tf.train.get_or_create_global_step()#重新定義global_step

optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(loss, global_step=global_step)

saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)#重新定義saver

# 定義研讀參數
training_epochs = 500  #設定迭代次數為500
display_step = 2

dataset = train_input_fn(train_data, batch_size)   #重復使用輸入函數train_input_fn
one_element = dataset.make_one_shot_iterator().get_next() #獲得輸入資料源張量  

with tf.Session() as sess:

    #還原估算器檢查點檔案
    savedir = "myestimatormode/"        
    kpt = tf.train.latest_checkpoint(savedir)       #找到檢查點檔案
    print("kpt:",kpt)
    saver.restore(sess, kpt)                       #還原檢查點資料
     
    # 向模型輸入資料
    while global_step.eval() < training_epochs:
        step = global_step.eval() 
        x,y =sess.run(one_element)

        sess.run(train_op, feed_dict={features: x, labels: y})

        #顯示訓練中的詳細訊息
        if step % display_step == 0:
            vloss = sess.run(loss, feed_dict={features: x, labels: y})
            print ("Epoch:", step+1, "cost=", vloss)
            if not (vloss == "NA" ):
                plotdata["batchsize"].append(global_step.eval())
                plotdata["loss"].append(vloss)
            saver.save(sess, savedir+"linermodel.cpkt", global_step)
                
    print (" Finished!")
    saver.save(sess, savedir+"linermodel.cpkt", global_step)
    
    print ("cost=", sess.run(loss,  feed_dict={features: x, labels: y}))


    
    plotdata["avgloss"] = moving_average(plotdata["loss"])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss')
     
    plt.show()

















