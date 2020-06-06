# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""
from sklearn.model_selection import train_test_split
import numpy as np
import os
import time
from PIL import Image
import tensorflow as tf

import matplotlib.pyplot as plt
tf.enable_eager_execution()



def make_dictionary():
    words_dic = [chr(i) for i in range(32,127)]
    words_dic.insert(0,'None')#補0用的
    words_dic.append("unknown")
    words_redic = dict(zip(words_dic, range(len(words_dic)))) #反向字典
    print('字表大小:', len(words_dic))
    return words_dic,words_redic


#字元到向量
def ch_to_v(datalist,words_redic,normal = 1):

    to_num = lambda word: words_redic[word] if word in words_redic else len(words_redic)-1# 字典裡沒有的就是None
    data_vector =[]
    for ii in datalist:
        data_vector.append(list(map(to_num, list(ii))))
    #歸一化
    if normal == 1:
        return np.asarray(data_vector)/ (len(words_redic)/2) - 1
    return np.array(data_vector)


def pad_sequences(sequences, maxlen=None, dtype=np.float32,
                  padding='post', truncating='post', value=0.):

    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x, lengths

#樣本資料預先處理（用於訓練）
def getbacthdata(batchx,charmap):
    batchx = ch_to_v( batchx,charmap,0)
    sampletpad ,sampletlengths =pad_sequences(batchx)#都填充為最大長度
    zero = np.zeros([len(batchx),1])
    tarsentence =np.concatenate((sampletpad[:,1:],zero),axis = 1)
    return np.asarray(sampletpad,np.int32),np.asarray(tarsentence,np.int32),sampletlengths

inv_charmap,charmap = make_dictionary()
vocab_size = len(charmap)

DATA_DIR ='./女孩名字.txt'  #定義載入的樣本路徑
input_text=[]
f = open(DATA_DIR)
import re
reforname=re.compile(r'[a-z]+', re.I)#用正則化，忽略大小寫分析字母
for i in f:
    t = re.match(reforname,i)
    if t:
        t=t.group()
        input_text.append(t)
        print(t)




input_text,target_text,sampletlengths = getbacthdata(input_text,charmap)
print(input_text)
print(target_text)

max_length = len(input_text[0])
learning_rate = 0.001

embedding_dim = 256#詞向量

units = 1024#GRU單元個數

BATCH_SIZE = 6#批次
#定義資料集
dataset = tf.data.Dataset.from_tensor_slices((input_text, target_text)).shuffle(1000)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)


class Model(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, units, batch_size):
    super(Model, self).__init__()
    self.units = units
    self.batch_sz = batch_size
    #定義詞內嵌層
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    if tf.test.is_gpu_available():#定義GRU cell
      self.gru = tf.keras.layers.CuDNNGRU(self.units,
                                          return_sequences=True,
                                          return_state=True,
                                          recurrent_initializer='glorot_uniform')
    else:
      self.gru = tf.keras.layers.GRU(self.units,
                                     return_sequences=True,
                                     return_state=True,
                                     recurrent_activation='sigmoid',
                                     recurrent_initializer='glorot_uniform')

    self.fc = tf.keras.layers.Dense(vocab_size)#定義全連線層

  def call(self, x, hidden):
    x = self.embedding(x)

    #使用gru網路進行計算，output的形狀為(batch_size, max_length, hidden_size)
    # states的形狀為(batch_size, hidden_size)
    output, states = self.gru(x, initial_state=hidden)

    #變換維度，用於後面的全連線，輸出形狀為 (batch_size * max_length, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    #得到每個詞的多項式分佈
    #輸出形狀為(max_length * batch_size, vocab_size)
    x = self.fc(output)
    return x, states

model = Model(vocab_size, embedding_dim, units, BATCH_SIZE)
optimizer = tf.train.AdamOptimizer()

#損失函數
def loss_function(real, preds):
    return tf.losses.sparse_softmax_cross_entropy(labels=real, logits=preds)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
latest_cpkt = tf.train.latest_checkpoint(checkpoint_dir)
if latest_cpkt:
    print('Using latest checkpoint at ' + latest_cpkt)
    checkpoint.restore(latest_cpkt)

else:
    os.makedirs(checkpoint_dir, exist_ok=True)


EPOCHS = 20


for epoch in range(EPOCHS):
    start = time.time()

    # initializing the hidden state at the start of every epoch
    hidden = model.reset_states()
    totaloss = []
    for (batch, (inp, target)) in enumerate(dataset):
          hidden = model.reset_states()
          with tf.GradientTape() as tape:
              # feeding the hidden state back into the model
              # This is the interesting step
              predictions, hidden = model(inp, hidden)

              # reshaping the target because that's how the
              # loss function expects it
              target = tf.reshape(target, (-1,))
              loss = loss_function(target, predictions)
              totaloss.append(loss)

          grads = tape.gradient(loss, model.variables)
          optimizer.apply_gradients(zip(grads, model.variables))

          if batch % 100 == 0:
              print ('Epoch {} Batch {} Loss {:.4f}'.format(epoch+1,
                                                            batch,
                                                            loss))
    # saving (checkpoint) the model every 5 epochs
    if (epoch + 1) % 2 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)



    print ('Epoch {} Loss {:.4f}'.format(epoch+1, np.mean(totaloss)))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Evaluation step(generating text using the model learned)


for iii in range(20):

    input_eval = input_text[np.random.randint(len(input_text))][0]
    start_string = inv_charmap[input_eval]
    input_eval = tf.expand_dims([input_eval], 0)
    #print(input_eval)



    # empty string to store our results
    text_generated = ''


    # hidden state shape == (batch_size, number of rnn units); here batch size == 1
    hidden = [tf.zeros((1, units))]
    #hidden = model.reset_states()
    for i in range(max_length):
        predictions, hidden = model(input_eval, hidden)

        predicted_id = tf.multinomial(predictions, num_samples=1)[0][0].numpy()
        if predicted_id==0:
            break

        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated += inv_charmap[predicted_id]

    print (start_string + text_generated)


