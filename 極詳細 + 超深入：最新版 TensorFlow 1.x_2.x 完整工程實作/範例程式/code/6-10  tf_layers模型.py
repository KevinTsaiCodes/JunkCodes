# -*- coding: utf-8 -*-
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
#print("Eager execution: {}".format(tf.executing_eagerly()))
#tf.logging.set_verbosity (tf.logging.ERROR)
# 載入訓練和驗證資料集

import tensorflow_datasets as tfds


ds_train, ds_test = tfds.load(name="mnist", split=["train", "test"]) #載入資料集
ds_train = ds_train.shuffle(1000).batch(10).prefetch(tf.data.experimental.AUTOTUNE)#用tf.data.Dataset接口加工資料集

class MNISTModel(tf.layers.Layer):
  def __init__(self, name):
    super(MNISTModel, self).__init__(name=name)

    self._input_shape = [-1, 28, 28, 1]
    self.conv1 =  tf.layers.Conv2D(32, 5,  activation=tf.nn.relu)
    self.conv2 =  tf.layers.Conv2D(64, 5,  activation=tf.nn.relu)
    self.fc1 = tf.layers.Dense(1024, activation=tf.nn.relu)
    self.fc2 = tf.layers.Dense(10)
    self.dropout = tf.layers.Dropout(0.5)
    self.max_pool2d =  tf.layers.MaxPooling2D(
            (2, 2), (2, 2), padding='SAME')

  def call(self, inputs, training):
    x = tf.reshape(inputs, self._input_shape)
    x = self.conv1(x)
    x = self.max_pool2d(x)
    x = self.conv2(x)
    x = self.max_pool2d(x)
    x = tf.keras.layers.Flatten()(x)
    x = self.fc1(x)
    if training:
      x = self.dropout(x)
    x = self.fc2(x)
    return x

def loss(model,inputs, labels):
    predictions = model(inputs, training=True)
    cost = tf.nn.sparse_softmax_cross_entropy_with_logits( logits=predictions, labels=labels )
    return tf.reduce_mean( cost )
# 訓練
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
grad = tfe.implicit_gradients(loss)


model = MNISTModel("net")

global_step = tf.train.get_or_create_global_step()

for epoch in range(1):
    for i,data in enumerate(ds_train):
        inputs, targets =tf.cast( data["image"],tf.float32), data["label"]

        optimizer.apply_gradients(grad( model,inputs, targets), global_step=global_step)

        if i % 400 == 0:
          print("Step %d: Loss on training set : %f" %
                (i, loss(model,inputs, targets).numpy()))


          all_variables = (
              model.variables
              + optimizer.variables()
              + [global_step])
          tfe.Saver(all_variables).save(
              "./log/linermodel.cpkt", global_step=global_step)
ds = tfds.as_numpy(ds_test.batch(100))
onetestdata = next(ds)

print("Loss on test set: %f" % loss( model,onetestdata["image"].astype(np.float32), onetestdata["label"]).numpy())
