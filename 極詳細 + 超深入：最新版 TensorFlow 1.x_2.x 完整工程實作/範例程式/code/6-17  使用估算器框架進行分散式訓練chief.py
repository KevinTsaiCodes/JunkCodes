# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室 
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""

TF_CONFIG='''{
    "cluster": {
        "chief": ["127.0.0.1:2221"],
        "worker": ["127.0.0.1:2222"],
        "ps": ["127.0.0.1:2223"]
    },
    "task": {"type": "chief", "index": 0}
}'''

import os
os.environ['TF_CONFIG']=TF_CONFIG
print(os.environ.get('TF_CONFIG'))


import tensorflow as tf
import numpy as np

#在記憶體中產生類比資料
def GenerateData(datasize = 100 ):
    train_X = np.linspace(-1, 1, datasize)   #train_X為-1到1之間連續的100個浮點數
    train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3 # y=2x，但是加入了噪聲
    return train_X, train_Y   #以產生器的模式傳回

train_data = GenerateData()  
test_data = GenerateData(20)  
batch_size=10

def train_input_fn(train_data, batch_size):  #定義訓練資料集輸入函數
    #建構資料集的群組成：一個特征輸入，一個標簽輸入
    dataset = tf.data.Dataset.from_tensor_slices( (  train_data[0],train_data[1]) )   
    dataset = dataset.shuffle(1000).repeat().batch(batch_size) #將資料集亂序、重復、批次劃分. 
    return dataset     #傳回資料集 

def eval_input_fn(data,labels, batch_size):  #定義測試或套用模型時，資料集的輸入函數
    #batch不容許為空
    assert batch_size is not None, "batch_size must not be None" 
    
    if labels is None:  #若果評估，則沒有標簽
        inputs = data  
    else:  
        inputs = (data,labels)  
    #建構資料集 
    dataset = tf.data.Dataset.from_tensor_slices(inputs)  
 
    dataset = dataset.batch(batch_size)  #按批次劃分
    return dataset     #傳回資料集     

def my_model(features, labels, mode, params):#自訂模型函數：參數是固定的。一個特征，一個標簽
    #定義網路結構
    W = tf.Variable(tf.random_normal([1]), name="weight")
    b = tf.Variable(tf.zeros([1]), name="bias")
    # 前嚮結構
    predictions = tf.multiply(tf.cast(features,dtype = tf.float32), W)+ b
    
    if mode == tf.estimator.ModeKeys.PREDICT: #預測處理
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    #定義損失函數
    loss = tf.losses.mean_squared_error(labels=labels, predictions=predictions)

    meanloss  = tf.metrics.mean(loss)#加入評估輸出項
    metrics = {'meanloss':meanloss}

    if mode == tf.estimator.ModeKeys.EVAL: #測試處理
        return tf.estimator.EstimatorSpec(   mode, loss=loss, eval_metric_ops=metrics) 
        #return tf.estimator.EstimatorSpec(   mode, loss=loss)

    #訓練處理.
    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)  


tf.reset_default_graph()  #清理圖
tf.logging.set_verbosity(tf.logging.INFO)      #能夠控制輸出訊息  ，
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)  #建構gpu_options，防止顯存占滿
session_config=tf.ConfigProto(gpu_options=gpu_options)
#建構估算器
estimator = tf.estimator.Estimator(  model_fn=my_model,model_dir='myestimatormode',params={'learning_rate': 0.1},
                                   config=tf.estimator.RunConfig(session_config=session_config)  )


train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(train_data, batch_size), max_steps=6000)
eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_input_fn(test_data,None, batch_size))

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)




##偏函數模式
#from functools import partial
#estimator.train(input_fn=partial(train_input_fn, train_data=train_data, batch_size=batch_size),steps=2)
#
##裝飾器模式
#def checkParams(fn):					#定義通用參數裝飾器函數
#    def wrapper():			#使用字典和元群組的解包參數來作形式參數
#        return fn(train_data=train_data, batch_size=batch_size)           	#如滿足條件，則將參數透傳給原函數，並傳回
#    return wrapper
#
#@checkParams
#def train_input_fn2(train_data, batch_size):  #定義訓練資料集輸入函數
#    #建構資料集的群組成：一個特征輸入，一個標簽輸入
#    dataset = tf.data.Dataset.from_tensor_slices( (  train_data[0],train_data[1]) )   
#    dataset = dataset.shuffle(1000).repeat().batch(batch_size) #將資料集亂序、重復、批次劃分. 
#    return dataset     #傳回資料集 
#estimator.train(input_fn=train_input_fn2, steps=2)
#
#tf.logging.info("訓練完成.")#輸出訓練完成




