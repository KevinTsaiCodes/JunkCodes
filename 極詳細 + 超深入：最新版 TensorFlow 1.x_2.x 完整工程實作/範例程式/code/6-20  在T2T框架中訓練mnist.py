# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

from tensor2tensor import problems
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import metrics

#啟動動態圖
tfe = tf.contrib.eager
tf.enable_eager_execution()

problems.available()



#建立路徑
data_dir = os.path.expanduser("./t2t/data")
tmp_dir = os.path.expanduser("./t2t/tmp")
tf.gfile.MakeDirs(data_dir)
tf.gfile.MakeDirs(tmp_dir)

#下載資料集
mnist_problem = problems.problem("image_mnist")
mnist_problem.generate_data(data_dir, tmp_dir)#下載到tmp_dir，（分為訓練和測試）轉換後存到data_dir

#取出一個資料，並顯示
Modes = tf.estimator.ModeKeys
mnist_example = tfe.Iterator(mnist_problem.dataset(Modes.TRAIN, data_dir)).next()
image = mnist_example["inputs"]#一個資料集元素的張量
label = mnist_example["targets"]

plt.imshow(image.numpy()[:, :, 0].astype(np.float32), cmap=plt.get_cmap('gray'))
print("Label: %d" % label.numpy())

print(Modes)

#registry.list_models()
#registry.list_hparams("transformer")


class MySimpleModel(t2t_model.T2TModel):#自訂模型

  def body(self, features):             #實現body方法
    inputs = features["inputs"]
    filters = self.hparams.hidden_size
    #h1=(in_width–filter_width + 1) / strides_ width =[12*12]
    h1 = tf.layers.conv2d(inputs, filters, kernel_size=(5, 5), strides=(2, 2))#預設valid
    #h2=[4*4]
    h2 = tf.layers.conv2d(tf.nn.relu(h1), filters, kernel_size=(5, 5), strides=(2, 2))

    return tf.layers.conv2d(tf.nn.relu(h2), filters, kernel_size=(3, 3))#[1*1]

hparams = trainer_lib.create_hparams("basic_1", data_dir=data_dir, problem_name="image_mnist")
hparams.hidden_size = 64
model = MySimpleModel(hparams, Modes.TRAIN)


#使用裝飾器implicit_value_and_gradients，來封裝loss函數
@tfe.implicit_value_and_gradients
def loss_fn(features):
  _, losses = model(features)
  return losses["training"]


BATCH_SIZE = 128        #指定批次
#建立資料集
mnist_train_dataset = mnist_problem.dataset(Modes.TRAIN, data_dir)
mnist_train_dataset = mnist_train_dataset.repeat(None).batch(BATCH_SIZE)

optimizer = tf.train.AdamOptimizer()#定義改善器



#訓練模型
NUM_STEPS = 500#指定訓練次數
for count, example in enumerate(mnist_train_dataset):
  example["targets"] = tf.reshape(example["targets"], [BATCH_SIZE, 1, 1, 1])  # Make it 4D.
  loss, gv = loss_fn(example)
  optimizer.apply_gradients(gv)

  if count % 50 == 0:
    print("Step: %d, Loss: %.3f" % (count, loss.numpy()))
  if count >= NUM_STEPS:
    break
#######
model.set_mode(Modes.EVAL)
mnist_eval_dataset = mnist_problem.dataset(Modes.EVAL, data_dir) #定義評估資料集

#建立評估metrics，傳回準確率與top5的準確率。
metrics_accum, metrics_result = metrics.create_eager_metrics(
    [metrics.Metrics.ACC, metrics.Metrics.ACC_TOP5])

for count, example in enumerate(mnist_eval_dataset):#檢查資料
  if count >= 200:#只取200個
    break

  #變化形狀
  example["inputs"] = tf.reshape(example["inputs"], [1, 28, 28, 1])
  example["targets"] = tf.reshape(example["targets"], [1, 1, 1, 1])

  predictions, _ = model(example)#用模型計算

  #計算統計值
  metrics_accum(predictions, example["targets"])

# Print out the averaged metric values on the eval data
for name, val in metrics_result().items():
  print("%s: %.2f" % (name, val))





















