# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""


import tensorflow as tf
import numpy as np
#tf.enable_eager_execution()
attention_keras = __import__("8-10  keras注意力機制模型")

#定義參數
num_words = 20000
maxlen = 80
batch_size = 32

#載入資料
print('Loading data...')
(x_train, y_train), (x_test, y_test) =  tf.keras.datasets.imdb.load_data(path='./imdb.npz',num_words=num_words)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print(x_train[0])
print(y_train[:10])

word_index = tf.keras.datasets.imdb.get_word_index('./imdb_word_index.json')# 單字--索引 對應字典
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])# 索引-單字對應字典

decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in x_train[0]])
print(decoded_newswire)



#資料對齊
x_train =  tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test =  tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
print('Pad sequences x_train shape:', x_train.shape)

#定義輸入節點
S_inputs = tf.keras.layers.Input(shape=(None,), dtype='int32')

#產生詞向量
embeddings = tf.keras.layers.Embedding(num_words, 128)(S_inputs)
embeddings = attention_keras.Position_Embedding()(embeddings) #預設使用同等維度的位置向量



#使用內定注意力模型處理
O_seq = attention_keras.Attention(8,16)([embeddings,embeddings,embeddings])
print("O_seq",O_seq)
#將結果進行全局池化
O_seq = tf.keras.layers.GlobalAveragePooling1D()(O_seq)
#加入dropout
#O_seq = tf.keras.layers.Dropout(0.5)(O_seq)
O_seq =attention_keras.TargetedDropout(drop_rate=0.5, target_rate=0.5)(O_seq)
#輸出最終節點
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(O_seq)
print(outputs)
#將網路結構群組合到一起
model = tf.keras.models.Model(inputs=S_inputs, outputs=outputs)

#加入反向傳播節點
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

#開始訓練
print('Train...')
model.fit(x_train, y_train, batch_size=batch_size,epochs=5, validation_data=(x_test, y_test))