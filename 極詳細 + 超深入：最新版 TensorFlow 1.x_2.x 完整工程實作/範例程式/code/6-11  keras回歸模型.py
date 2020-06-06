# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""

import tensorflow as tf
import numpy as np
import os

#在記憶體中產生類比資料
def GenerateData(datasize = 100 ):
    train_X = np.linspace(-1, 1, datasize)   #train_X為-1到1之間連續的100個浮點數
    train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3 # y=2x，但是加入了噪聲
    return train_X, train_Y   #以產生器的模式傳回

train_data = GenerateData()

#直接使用model定義網路
inputs = tf.keras.Input(shape=(1,))
outputs= tf.keras.layers.Dense(1)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

#定義2層網路
x = tf.keras.layers.Dense(1, activation='tanh')(inputs)
outputs_2 = tf.keras.layers.Dense(1)(x)
model_2 = tf.keras.Model(inputs=inputs, outputs=outputs_2)



#使用sequential 指定input的形狀
model_3 = tf.keras.models.Sequential()
model_3.add(tf.keras.layers.Dense(1, input_shape=(1,)))
model_3.add(tf.keras.layers.Dense(units = 1))



#使用sequential 指定帶批次的input形狀
model_4 = tf.keras.models.Sequential()
model_4.add(tf.keras.layers.Dense(1, batch_input_shape=(None, 1)))

#使用sequential 指定input的維度
model_5 = tf.keras.models.Sequential()
model_5.add(tf.keras.layers.Dense( 1, input_dim = 1))

#使用sequential 預設input
model_6 = tf.keras.models.Sequential()
model_6.add(tf.keras.layers.Dense(1))
print(model_6.weights)
model_6.build((None, 1))#指定輸入，開始產生模型
print(model_6.weights)





# 選取損失函數和改善方法
model.compile(loss = 'mse', optimizer = 'sgd')
model_3.compile(loss = tf.losses.mean_squared_error, optimizer = 'sgd')

# 進行訓練, 傳回損失(代價)函數
for step in range(201):
    cost = model.train_on_batch(train_data[0], train_data[1])
    if step % 10 == 0:
        print ('loss: ', cost)




#直接使用fit函數來訓練
model_3.fit(x=train_data[0],y=train_data[1], batch_size=10, epochs=20)

# 取得參數
W,b= model.get_weights()
print ('Weights: ',W)
print ('Biases: ', b)

#對於 使用sequential的模型可以指定實際層來取得
W, b = model_3.layers[0].get_weights()
print ('Weights: ',W)
print ('Biases: ', b)



cost = model.evaluate(train_data[0], train_data[1], batch_size = 10)#測試
print ('test loss: ', cost)

a = model.predict(train_data[0], batch_size = 10)#預測
print(a[:10])
print(train_data[1][:10])


#儲存及載入模型
model.save('my_model.h5')

del model  #移除目前模型
#載入
model = tf.keras.models.load_model('my_model.h5')

a = model.predict(train_data[0], batch_size = 10)
print("載入後的測試",a[:10])

#產生tf格式模型
model.save_weights('./keraslog/kerasmodel') #若果是以 '.h5'或'.keras'結尾，預設會產生keras格式模型

#產生tf格式模型，手動指定
os.makedirs("./kerash5log", exist_ok=True)
model.save_weights('./kerash5log/kerash5model',save_format = 'h5')#可以指定save_format為h5 或tf來產生對應的格式


json_string = model.to_json()  #同等於 json_string = model.get_config()
open('my_model.json','w').write(json_string)

#載入模型資料和weights
model_7 = tf.keras.models.model_from_json(open('my_model.json').read())
model_7.load_weights('my_model.h5')
a = model_7.predict(train_data[0], batch_size = 10)
print("載入後的測試",a[:10])

import h5py
f=h5py.File('my_model.h5')
for name in f:
    print(name)
