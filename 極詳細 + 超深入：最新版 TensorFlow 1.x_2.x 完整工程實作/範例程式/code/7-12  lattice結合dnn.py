# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室 
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""

import os
import tempfile
import numpy as np

import pandas as pd
import six
import tensorflow as tf
import tensorflow_lattice as tfl



#定義資料集目錄.
testdir = "./income_data/adult.test.csv.txt"
traindir = "./income_data/adult.data.csv.txt"


batch_size = 1000 #定義批次
hparams = None  # --hparams=learning_rate=0.1,lattice_size=2,num_keypoints=100"


#定義列名，對應csv中的資料列.
CSV_COLUMNS = [
    "age", "workclass", "fnlwgt",
    "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "gender",
    "capital_gain", "capital_loss", "hours_per_week", "native_area",
    "income_bracket"
]


_df_data = {}       #以字典形式，存放csv檔名和對應的樣本內容
_df_data_labels = {} #以字典形式，存放csv檔名和對應的標簽內容

#讀入原始csv檔案，並轉成估算器的輸入函數
def get_input_fn(file_path, batch_size, num_epochs, shuffle):

  if file_path not in _df_data: #確保只讀取依次csv
    #讀取csv檔案，並將樣本內容放入_df_data中
    _df_data[file_path] = pd.read_csv( tf.gfile.Open(file_path),
        names=CSV_COLUMNS,skipinitialspace=True,
        engine="python", skiprows=1)
    
    _df_data[file_path] = _df_data[file_path].dropna(how="any", axis=0)
    #讀取csv檔案，並將標簽內容放入_df_data_labels中
    _df_data_labels[file_path] = _df_data[file_path]["income_bracket"].apply(
        lambda x: ">50K" in x).astype(int)
    
  return tf.estimator.inputs.pandas_input_fn( #傳回pandas結構的輸入函數
      x=_df_data[file_path],y=_df_data_labels[file_path],
      batch_size=batch_size,shuffle=shuffle,
      num_epochs=num_epochs,num_threads=1)

def create_feature_columns():#建立特征列
  #離雜湊.
  gender = tf.feature_column.categorical_column_with_vocabulary_list(
      "gender", ["Female", "Male"])
  education = tf.feature_column.categorical_column_with_vocabulary_list(
      "education", [
          "Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college",
          "Assoc-acdm", "Assoc-voc", "7th-8th", "Doctorate", "Prof-school",
          "5th-6th", "10th", "1st-4th", "Preschool", "12th"
      ])
  marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
      "marital_status", [
          "Married-civ-spouse", "Divorced", "Married-spouse-absent",
          "Never-married", "Separated", "Married-AF-spouse", "Widowed"
      ])
  relationship = tf.feature_column.categorical_column_with_vocabulary_list(
      "relationship", [
          "Husband", "Not-in-family", "Wife", "Own-child", "Unmarried",
          "Other-relative"
      ])
  workclass = tf.feature_column.categorical_column_with_vocabulary_list(
      "workclass", [
          "Self-emp-not-inc", "Private", "State-gov", "Federal-gov",
          "Local-gov", "?", "Self-emp-inc", "Without-pay", "Never-worked"
      ])
  occupation = tf.feature_column.categorical_column_with_vocabulary_list(
      "occupation", [
          "Prof-specialty", "Craft-repair", "Exec-managerial", "Adm-clerical",
          "Sales", "Other-service", "Machine-op-inspct", "?",
          "Transport-moving", "Handlers-cleaners", "Farming-fishing",
          "Tech-support", "Protective-serv", "Priv-house-serv", "Armed-Forces"
      ])
  race = tf.feature_column.categorical_column_with_vocabulary_list(
      "race", [ "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo",
               "Other",]   )
  native_area = tf.feature_column.categorical_column_with_vocabulary_list(
      "native_area", ["area_A","area_B","?", "area_C",
          "area_D", "area_E", "area_F","area_G","area_H","area_I",
          "Greece", "area_K","area_L","area_M","area_N","area_O",
          "area_P","Italy","area_R", "Jamaica","area_T","Mexico","area_S",
          "area_U","France","area_W","area_V","Ecuador","area_X", "Columbia",
          "area_Y", "Guatemala","Nicaragua","area_Z", "area_1A",
          "area_1B", "area_1C","area_1D","Peru",
          "area_#", "area_1G",])
     

  #連續值列.
  age = tf.feature_column.numeric_column("age")
  education_num = tf.feature_column.numeric_column("education_num")
  capital_gain = tf.feature_column.numeric_column("capital_gain")
  capital_loss = tf.feature_column.numeric_column("capital_loss")
  hours_per_week = tf.feature_column.numeric_column("hours_per_week")
  
  dnnfeature = [age,education_num,capital_gain,capital_loss,hours_per_week,
           tf.feature_column.indicator_column(gender), 
           tf.feature_column.indicator_column(education), 
           tf.feature_column.indicator_column(marital_status), 
           tf.feature_column.indicator_column(relationship), 
           tf.feature_column.indicator_column(workclass), 
           tf.feature_column.indicator_column(occupation), 
           tf.feature_column.indicator_column(race), 
           tf.feature_column.indicator_column(native_area),            
          ]
  
  #將處理好的特征列傳回。fnlwgt不需要，income-bracket是標簽，也不需要
  return [ age, workclass, education, education_num, marital_status,
      occupation, relationship,race,gender, capital_gain,
      capital_loss, hours_per_week, native_area,],dnnfeature


#建立校准關鍵點
def create_quantiles(quantiles_dir):
  batch_size = 10000 #設定批次
  _,fc = create_feature_columns()
  
  #建立輸入函數
  input_fn =get_input_fn(traindir, batch_size, num_epochs=1, shuffle=False)
  
  tfl.save_quantiles_for_keypoints( #預設儲存1000個校准關鍵點
      input_fn=input_fn,
      save_dir=quantiles_dir, #預設會建立一個檔案目錄
      feature_columns=fc,
      num_steps=None)
  

quantiles_dir = "./dnnquant"
create_quantiles(quantiles_dir) #建立校准關鍵點訊息
a = tfl.load_keypoints_from_quantiles(["age"],quantiles_dir,num_keypoints=10,
                                  output_min=17.0,output_max=90.0,)
with tf.Session() as sess:
    print("載入age的關鍵點訊息：",sess.run(a)) 
    
    #{'age': (array([17., 25., 33., 41., 49., 57., 65., 73., 81., 90.], dtype=float32), array([17.      , 25.11111 , 33.22222 , 41.333332, 49.444443, 57.555553,
#       65.666664, 73.77777 , 81.888885, 90.      ], dtype=float32))} 


#####################################################

#輸出超參。用於顯示
def _pprint_hparams(hparams):
  print("* hparams=[")
  for (key, value) in sorted(six.iteritems(hparams.values())):
    print("\t{}={}".format(key, value))
  print("]")




#建立calibrated_dnn模型
def create_calibrated_dnn(feature_columns, config, quantiles_dir):
  """Creates a calibrated DNN model."""
  # This is an example of a hybrid model that uses input calibration layer
  # offered by TensorFlow Lattice library and connects it to a DNN.
  feature_names = [fc.name for fc in feature_columns[1]]
  print(feature_names)
  print([fc.name for fc in feature_columns[0]])
  hparams = tfl.CalibratedHParams(feature_names=feature_names,
      num_keypoints=200,learning_rate=1.0e-3,calibration_output_min=-1.0,
      calibration_output_max=1.0,
      nodes_per_layer=10,  #每層10個節點
      layers=2,  #內含輸出層，一共2層
  )
  #hparams.parse(hparams)
  _pprint_hparams(hparams)


  #建構含有點陣的神經網路模型
  def _model_fn(features, labels, mode, params):
    #del params  #移除不需要的params，直接取外部的超參hparams
    #_pprint_hparams(hparams)
    hparams = params
    _pprint_hparams(hparams)

    
#    sorted(feature_columns, key=lambda fc : fc.name)
#    
#    net = tf.feature_column.input_layer(features,feature_columns[1])
    a = {}
    for fc in feature_columns[1]:
        a[fc.name]= tf.feature_column.input_layer(features,[fc])

        

    #使用超參為輸入建立一個校准層，中間兩個參數為列名清單、強制單調時，對數值映射的op清單
    #因為作為中間層，所以不需要關心。
    (output, _, _, regularization) = tfl.input_calibration_layer_from_hparams(
         a, hparams, quantiles_dir)

    #全連線隱藏層.
    for _ in range(hparams.layers - 1):
      output = tf.layers.dense(
          inputs=output, units=hparams.nodes_per_layer, activation=tf.sigmoid)

    #最終分類別的輸出層.
    logits = tf.layers.dense(inputs=output, units=1)
    predictions = tf.reshape(tf.sigmoid(logits), [-1])

    #計算loss
    loss_no_regularization = tf.losses.log_loss(labels, predictions)
    loss = loss_no_regularization
    if regularization is not None:#在loss中，加入點陣層的正則
      loss += regularization
    optimizer = tf.train.AdamOptimizer(learning_rate=hparams.learning_rate)
    train_op = optimizer.minimize(
        loss,
        global_step=tf.train.get_global_step(),
        name="calibrated_dnn_minimize")

    eval_metric_ops = {  #用於輸出中間結果
        "accuracy": tf.metrics.accuracy(labels, predictions),
        "average_loss": tf.metrics.mean(loss_no_regularization),
    }

    return tf.estimator.EstimatorSpec(mode, predictions, loss, train_op,
                                      eval_metric_ops)

  #呼叫建構好的模型，產生估算器
  return tf.estimator.Estimator(
      model_fn=_model_fn,
      model_dir=config.model_dir,
      config=config,
      params=hparams)




#用指定資料測試模型
def evaluate_on_data(estimator, data):
  name = os.path.basename(data) #或取輸入資料的資料夾名稱
  
  #評估模型
  evaluation = estimator.evaluate(input_fn=get_input_fn(  #定義輸入函數
     file_path=data, batch_size=batch_size,num_epochs=1,shuffle=False),
                                  name=name)
  print("  Evaluation on '{}':\t準確率={:.4f}\t平均loss={:.4f}".format(
      name, evaluation["accuracy"], evaluation["average_loss"]))

def evaluate(estimator):#分別用訓練和測試資料集，對模型進行測試 
  evaluate_on_data(estimator, traindir)
  evaluate_on_data(estimator, testdir)

def train(estimator,train_epochs,showtest = None):
  if showtest==None:  #不顯示中間測試訊息
    input_fn =get_input_fn(traindir, batch_size, num_epochs=train_epochs, shuffle=True)
    estimator.train(input_fn=input_fn)
  else:#訓練過程中，顯示10次中間模型的測試訊息.
    epochs_trained = 0
    loops = 0
    while epochs_trained < train_epochs:
      loops += 1
      next_epochs_trained = int(loops * train_epochs / 10.0)
      epochs = max(1, next_epochs_trained - epochs_trained)
      epochs_trained += epochs
      input_fn =get_input_fn(traindir, batch_size, num_epochs=epochs, shuffle=True)
      estimator.train(input_fn=input_fn)
      print("Trained for {} epochs, total so far {}:".format(
          epochs, epochs_trained))
      evaluate(estimator)



allfeature_columns = create_feature_columns() #建立特征列

modelsfun = [create_calibrated_dnn]#建立calibrated_dnn模型函數

for modelfun in modelsfun: #依次建立函數，對其評估
    print('{0:-^50}'.format(modelfun.__name__))#分隔符
    
    output_dir = "./model_"+modelfun.__name__
    os.makedirs(output_dir, exist_ok=True)#建立模型路徑
    #建立估算器組態檔
    config = tf.estimator.RunConfig().replace(model_dir=output_dir)
    
    estimator = modelfun(allfeature_columns,config, quantiles_dir)#建立估算器
    train(estimator,train_epochs=10)#訓練模型,迭代10次
    evaluate(estimator)#評估模型

