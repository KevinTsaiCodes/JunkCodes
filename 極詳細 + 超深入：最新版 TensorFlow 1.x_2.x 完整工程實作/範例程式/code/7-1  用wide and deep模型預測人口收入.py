


import argparse               #引入系統頭檔案
import os
import shutil
import sys

import tensorflow as tf       #引TensorFlow入頭檔案
from utils import parsers,hooks_helper,model_helpers  #引入utils頭檔案


_CSV_COLUMNS = [                                #定義CVS列名
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_area',
    'income_bracket'
]

_CSV_COLUMN_DEFAULTS = [                        #定義每一列的預設值
        [0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]

_NUM_EXAMPLES = {                               #定義樣本集的數量
    'train': 32561,
    'validation': 16281,
}


LOSS_PREFIX = {'wide': 'linear/', 'deep': 'dnn/'}#定義模型的前綴


def build_model_columns():
  """產生wide和deep模型的特征列集合."""
  #定義連續值列
  age = tf.feature_column.numeric_column('age')
  education_num = tf.feature_column.numeric_column('education_num')
  capital_gain = tf.feature_column.numeric_column('capital_gain')
  capital_loss = tf.feature_column.numeric_column('capital_loss')
  hours_per_week = tf.feature_column.numeric_column('hours_per_week')

  #定義離散值列，傳回的是稀疏矩陣
  education = tf.feature_column.categorical_column_with_vocabulary_list(
      'education', [
          'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
          'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
          '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

  marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
      'marital_status', [
          'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
          'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

  relationship = tf.feature_column.categorical_column_with_vocabulary_list(
      'relationship', [
          'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
          'Other-relative'])

  workclass = tf.feature_column.categorical_column_with_vocabulary_list(
      'workclass', [
          'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
          'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])

  #將所有職業名稱透過hash算法，離散成1000個類別別:
  occupation = tf.feature_column.categorical_column_with_hash_bucket(
      'occupation', hash_bucket_size=1000)

  #將連續值特征列轉為離散值特征.
  age_buckets = tf.feature_column.bucketized_column(
      age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

  #定義基礎特征列.
  base_columns = [
      education, marital_status, relationship, workclass, occupation,
      age_buckets,
  ]
  #定義交叉特征列.
  crossed_columns = [
      tf.feature_column.crossed_column(
          ['education', 'occupation'], hash_bucket_size=1000),
      tf.feature_column.crossed_column(
          [age_buckets, 'education', 'occupation'], hash_bucket_size=1000),
  ]

  #定義wide模型的特征列.
  wide_columns = base_columns + crossed_columns

  #定義deep模型的特征列.
  deep_columns = [
      age,
      education_num,
      capital_gain,
      capital_loss,
      hours_per_week,
      tf.feature_column.indicator_column(workclass),              #將workclass列的稀疏矩陣轉成0ne_hot解碼
      tf.feature_column.indicator_column(education),
      tf.feature_column.indicator_column(marital_status),
      tf.feature_column.indicator_column(relationship),
      tf.feature_column.embedding_column(occupation, dimension=8),#將1000個hash後的類別別，每個用內嵌詞embedding轉換
  ]

  return wide_columns, deep_columns


def build_estimator(model_dir, model_type,warm_start_from=None):
  """按照特殊的模型產生估算器物件."""
  wide_columns, deep_columns = build_model_columns()
  hidden_units = [100, 75, 50, 25]

  run_config = tf.estimator.RunConfig().replace(                #將GPU個數設為0，關閉GPU運算。因為該模型在CPU上速度更快
      session_config=tf.ConfigProto(device_count={'GPU': 0}),
      save_checkpoints_steps=1000)

  if model_type == 'wide':                                      #產生帶有wide模型的估算器物件
    return tf.estimator.LinearClassifier(
        model_dir=model_dir,
        feature_columns=wide_columns,
        config=run_config)
  elif model_type == 'deep':                                    #產生帶有deep模型的估算器物件
    return tf.estimator.DNNClassifier(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=hidden_units,
        config=run_config)
  else:
    return tf.estimator.DNNLinearCombinedClassifier(            #產生帶有wide和deep模型的估算器物件
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        config=run_config,
        warm_start_from=warm_start_from)


def input_fn(data_file, num_epochs, shuffle, batch_size):       #定義估算器輸入函數
  """估算器的輸入函數."""
  assert tf.gfile.Exists(data_file), (                          #用斷言敘述判斷樣本檔案是否存在
      '%s not found. Please make sure you have run data_download.py and '
      'set the --data_dir argument to the correct path.' % data_file)

  def parse_csv(value):
    print('Parsing', data_file)
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
    features = dict(zip(_CSV_COLUMNS, columns))
    labels = features.pop('income_bracket')
    return features, tf.equal(labels, '>50K')


  dataset = tf.data.TextLineDataset(data_file)                  #建立dataset資料集

  if shuffle:                                                   #對資料進行亂序動作
    dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

  dataset = dataset.map(parse_csv, num_parallel_calls=5)        #對data_file檔案中的每行資料，進行特征抽取，傳回新的資料集

  dataset = dataset.repeat(num_epochs)                          #將資料集重復num_epochs次
  dataset = dataset.batch(batch_size)                           #將資料集按照batch_size劃分
  dataset = dataset.prefetch(1)
  return dataset


def export_model(model, model_type, export_dir):                #定義函數export_model ，用於匯出模型
  """匯出模型.

  參數:
    model: 估算器物件
    model_type: 要匯出的模型型態，可選值有 "wide"、 "deep" 或 "wide_deep"
    export_dir: 匯出模型的路徑.
  """
  wide_columns, deep_columns = build_model_columns()        #獲得列張量
  if model_type == 'wide':
    columns = wide_columns
  elif model_type == 'deep':
    columns = deep_columns
  else:
    columns = wide_columns + deep_columns
  feature_spec = tf.feature_column.make_parse_example_spec(columns)
  example_input_fn = (
      tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec))
  model.export_savedmodel(export_dir, example_input_fn)


class WideDeepArgParser(argparse.ArgumentParser):               #定義WideDeepArgParser類別，用於解析參數
  """該類別用於在程式啟動時的參數解析."""

  def __init__(self):                                           #起始化函數
    super(WideDeepArgParser, self).__init__(parents=[parsers.BaseParser()]) #呼叫父類別的起始化函數
    self.add_argument(
        '--model_type', '-mt', type=str, default='wide_deep',   #加入一個啟動參數--model_type，預設值為wide_deep
        choices=['wide', 'deep', 'wide_deep'],                  #定義該參數的可選值
        help='[default %(default)s] Valid model types: wide, deep, wide_deep.', #定義啟動參數的幫助指令
        metavar='<MT>')
    self.set_defaults(                                          #為其他參數設定預設值
        data_dir='income_data',                                 #設定資料樣本路徑
        model_dir='income_model',                               #設定模型存放路徑
        export_dir='income_model_exp',                          #設定匯出模型存放路徑
        train_epochs=5,                                        #設定迭代次數
        batch_size=40)                                          #設定批次大小

def trainmain(argv):
  parser = WideDeepArgParser()                                  #案例化WideDeepArgParser，用於解析啟動參數
  flags = parser.parse_args(args=argv[1:])                      #獲得解析後的參數flags
  print("解析的參數為：",flags)

  shutil.rmtree(flags.model_dir, ignore_errors=True)            #若果模型存在，整個目錄移除
  model = build_estimator(flags.model_dir, flags.model_type)    #產生估算器物件

  train_file = os.path.join(flags.data_dir, 'adult.data.csv')       #獲得訓練集樣本檔案的路徑
  test_file = os.path.join(flags.data_dir, 'adult.test.csv')        #獲得測試集樣本檔案的路徑


  def train_input_fn():                                         #定義訓練集樣本輸入函數
    return input_fn(                                            #該輸入函數按照batch_size批次,迭代輸入epochs_between_evals次，使用亂序處理
        train_file, flags.epochs_between_evals, True, flags.batch_size)

  def eval_input_fn():                                          #定義測試集樣本輸入函數
    return input_fn(test_file, 1, False, flags.batch_size)      #該輸入函數按照batch_size批次,迭代輸入1次，不使用亂序處理

  loss_prefix = LOSS_PREFIX.get(flags.model_type, '')           #格式化輸出loss的前綴

  train_hook = hooks_helper.get_logging_tensor_hook(                   #定義訓練鉤子，獲得訓練過程中的狀態
      batch_size=flags.batch_size,
      tensors_to_log={'average_loss': loss_prefix + 'head/truediv',
                      'loss': loss_prefix + 'head/weighted_loss/Sum'})
  for n in range(flags.train_epochs ): #將總迭代數，按照epochs_between_evals分段。並循環對每段進行訓練
    model.train(input_fn=train_input_fn, hooks=[train_hook])         #呼叫估算器的train方法進行訓練
    results = model.evaluate(input_fn=eval_input_fn)                #呼叫估算器的evaluate方法進行評估計算

    print('{0:-^60}'.format('evaluate at epoch %d'%( (n + 1))))#分隔符

    for key in sorted(results):                                     #顯示評估結果
      print('%s: %s' % (key, results[key]))

    if model_helpers.past_stop_threshold(                           #根據accuracy的設定值，來判斷是否需要結束訓練。
        flags.stop_threshold, results['accuracy']):
      break

  if flags.export_dir is not None:                                  #根據設定匯出凍結圖模型，用於tfseving
    export_model(model, flags.model_type, flags.export_dir)




def premain(argv):
  parser = WideDeepArgParser()                                  #案例化WideDeepArgParser，用於解析啟動參數
  flags = parser.parse_args(args=argv[1:])                      #獲得解析後的參數flags
  print("解析的參數為：",flags)

  test_file = os.path.join(flags.data_dir, 'adult.test.csv')        #獲得測試集樣本檔案的路徑

  def eval_input_fn():                                          #定義測試集樣本輸入函數
    return input_fn(test_file, 1, False, flags.batch_size)      #該輸入函數按照batch_size批次,迭代輸入1次，不使用亂序處理

#  model2 = build_estimator('./temp', flags.model_type,flags.model_dir)#也可以使用熱啟動的模式
  model2 = build_estimator(flags.model_dir, flags.model_type)

  predictions = model2.predict(input_fn=eval_input_fn)
  for i, per in enumerate(predictions):
      print("csv中第",i,"條結果為：",per['class_ids'])
      if i==5:
          break

if __name__ == '__main__':                  #當執行檔案時，模組名字__name__就會為__main__
  tf.logging.set_verbosity(tf.logging.ERROR) #設定log等級為INFO，若果想要顯示的訊息少點，可以設定成 WARN
  trainmain(argv=sys.argv)                       #呼叫main函數，進入程式主體
  premain(argv=sys.argv)
