# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""

from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.resnet50 import preprocess_input, decode_predictions

import numpy as np
#6.7.10節 動態圖
#tf.enable_eager_execution()						#啟動動態圖

model = ResNet50(weights='imagenet')
#model = ResNet50(weights='resnet50_weights_tf_dim_ordering_tf_kernels.h5')


img_path = 'hy.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])

#6.7.10節 動態圖
#preds = model(x)
#print('Predicted:', decode_predictions(preds.numpy(), top=3)[0])