# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""



#使用動態圖訓練一個具有檢查點的回歸模型

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()
print("TensorFlow 版本: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

#（1）產生類比資料
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3 # y=2x，但是加入了噪聲
#圖形顯示
plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.legend()
plt.show()

# 定義研讀參數
W = tf.Variable(tf.random_normal([1]),dtype=tf.float32, name="weight")
b = tf.Variable(tf.zeros([1]),dtype=tf.float32, name="bias")

global_step = tf.train.get_or_create_global_step()

def getcost(x,y):#定義函數，計算loss值
    # 前嚮結構
    z = tf.cast(tf.multiply(np.asarray(x,dtype = np.float32), W)+ b,dtype = tf.float32)
    cost =tf.reduce_mean( tf.square(y - z))#loss值
    return cost

learning_rate = 0.01
# 隨機梯度下降法作為改善器
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

grad = tfe.implicit_gradients(getcost)#獲得計算梯度的函數


#定義saver，示範兩種方法處理檢查點檔案
savedir = "logeager/"
savedirx = "logeagerx/"

saver = tf.train.Saver([W,b], max_to_keep=1)#產生saver。 max_to_keep=1，表明最多只儲存一個檢查點檔案
saverx = tfe.Saver([W,b])#產生saver。 max_to_keep=1，表明最多只儲存一個檢查點檔案



kpt = tf.train.latest_checkpoint(savedir)#找到檢查點檔案
kptx = tf.train.latest_checkpoint(savedirx)#找到檢查點檔案
if kpt!=None:
    saver.restore(None, kpt) #兩種載入模式都可以
    saverx.restore(kptx)

training_epochs = 20  #迭代訓練次數
display_step = 2

plotdata = { "batchsize":[], "loss":[] }#收集訓練參數

while global_step/len(train_X) < training_epochs: #迭代訓練模型
    step = int( global_step/len(train_X) )
    for (x, y) in zip(train_X, train_Y):
        optimizer.apply_gradients(grad(x, y),global_step)

    #顯示訓練中的詳細訊息
    if step % display_step == 0:
        cost = getcost (x, y)
        print ("Epoch:", step+1, "cost=", cost.numpy(),"W=", W.numpy(), "b=", b.numpy())
        if not (cost == "NA" ):
            plotdata["batchsize"].append(global_step.numpy())
            plotdata["loss"].append(cost.numpy())
        saver.save(None, savedir+"linermodel.cpkt", global_step)
        saverx.save(savedirx+"linermodel.cpkt", global_step)


print (" Finished!")
saver.save(None, savedir+"linermodel.cpkt", global_step)
saverx.save(savedirx+"linermodel.cpkt", global_step)
print ("cost=", getcost (train_X, train_Y).numpy() , "W=", W.numpy(), "b=", b.numpy())

#顯示模型
plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.plot(train_X, W * train_X + b, label='Fitted line')
plt.legend()
plt.show()

def moving_average(a, w=10):#定義產生loss可視化的函數
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

plotdata["avgloss"] = moving_average(plotdata["loss"])
plt.figure(1)
plt.subplot(211)
plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
plt.xlabel('Minibatch number')
plt.ylabel('Loss')
plt.title('Minibatch run vs. Training loss')

plt.show()
