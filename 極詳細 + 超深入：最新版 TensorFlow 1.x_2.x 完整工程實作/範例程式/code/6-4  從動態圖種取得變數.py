
"""
@author: 程式碼醫生工作室
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""

import tensorflow as tf
import numpy as np
import tensorflow.contrib.eager as tfe


tf.enable_eager_execution()
print("TensorFlow 版本: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

#產生類比資料
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3 # y=2x，但是加入了噪聲

#建立資料集
dataset = tf.data.Dataset.from_tensor_slices( (np.reshape(train_X,[-1,1]),np.reshape(train_X,[-1,1])) )
dataset = dataset.repeat().batch(1)
global_step = tf.train.get_or_create_global_step()
container = tfe.EagerVariableStore()
learning_rate = 0.01
# 隨機梯度下降法作為改善器
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

def getcost(x,y):#定義函數，計算loss值
    # 前嚮結構
    with container.as_default():#將動態圖使用的層包裝起來。可以得到變數

#        z = tf.contrib.slim.fully_connected(x, 1,reuse=tf.AUTO_REUSE)
        z = tf.layers.dense(x,1, name="l1")
    cost =tf.reduce_mean( tf.square(y - z))#loss值
    return cost

def grad( inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = getcost(inputs, targets)
    return tape.gradient(loss_value,container.trainable_variables())

training_epochs = 20  #迭代訓練次數
display_step = 2

#迭代訓練模型
for step,value in enumerate(dataset) :
    grads = grad( value[0], value[1])
    optimizer.apply_gradients(zip(grads, container.trainable_variables()), global_step=global_step)
    if step>=training_epochs:
        break

    #顯示訓練中的詳細訊息
    if step % display_step == 0:
        cost = getcost (value[0], value[1])
        print ("Epoch:", step+1, "cost=", cost.numpy())

print (" Finished!")
print ("cost=", cost.numpy() )
for i in container.trainable_variables():
    print(i.name,i.numpy())


