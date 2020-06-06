# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室 
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""
import tensorflow as tf
import sys                                      
nets_path = r'slim'                             #載入環境變數
if nets_path not in sys.path:
    sys.path.insert(0,nets_path)
else:
    print('already add slim')
from nets.nasnet import nasnet                 #匯出nasnet
slim = tf.contrib.slim                         #slim
image_size = nasnet.build_nasnet_mobile.default_image_size  #獲得圖片輸入尺寸 224
from preprocessing import preprocessing_factory#圖形處理


import os
def list_images(directory):
    """
    取得所有directory中的所有圖片和標簽
    """

    #傳回path特殊的資料夾包括的檔案或資料夾的名字的清單
    labels = os.listdir(directory)
    #對標簽進行排序，以便訓練和驗證按照相同的順序進行
    labels.sort()
    #建立檔案標簽清單
    files_and_labels = []
    for label in labels:
        for f in os.listdir(os.path.join(directory, label)):
            #轉換字串中所有大寫字元為小寫再判斷
            if 'jpg' in f.lower() or 'png' in f.lower():
                #加入清單
                files_and_labels.append((os.path.join(directory, label, f), label))
    #瞭解為解壓 把資料路徑和標簽解壓出來
    filenames, labels = zip(*files_and_labels)
    #轉為清單 分別儲存資料路徑和對應標簽
    filenames = list(filenames)
    labels = list(labels)
    #列出分類別總數 例如兩類別：['man', 'woman']
    unique_labels = list(set(labels))

    label_to_int = {}
    #循環列出資料和資料索引
    #給每個分類別打上標簽{'woman': 2, 'man': 1，none：0}
    for i, label in enumerate(sorted(unique_labels)):
        label_to_int[label] = i+1
    print(label,label_to_int[label])
    #把每個標簽化為0 1 這種形式
    labels = [label_to_int[l] for l in labels]
    print(labels[:6],labels[-6:])
    return filenames, labels  #傳回儲存資料路徑和對應轉換後的標簽


num_workers = 2  #定義平行處理資料的執行緒數量

#圖形批次預先處理
image_preprocessing_fn = preprocessing_factory.get_preprocessing('nasnet_mobile', is_training=True)
image_eval_preprocessing_fn = preprocessing_factory.get_preprocessing('nasnet_mobile', is_training=False)

def _parse_function(filename, label):  #定義圖形解碼函數
    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)          
    return image, label

def training_preprocess(image, label):  #定義調整圖形大小函數
    image = image_preprocessing_fn(image, image_size, image_size)
    return image, label

def val_preprocess(image, label):   #定義評估圖形預先處理函數
    image = image_eval_preprocessing_fn(image, image_size, image_size)
    return image, label

#建立帶批次的資料集
def creat_batched_dataset(filenames, labels,batch_size,isTrain = True):
    
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    dataset = dataset.map(_parse_function, num_parallel_calls=num_workers)#對圖形解碼
        
    if isTrain == True:
        dataset = dataset.shuffle(buffer_size=len(filenames))#打亂資料順序
        dataset = dataset.map(training_preprocess, num_parallel_calls=num_workers)#調整圖形大小
    else:
        dataset = dataset.map(val_preprocess,num_parallel_calls=num_workers)#調整圖形大小
        
    return dataset.batch(batch_size)   #傳回批次資料

#根據目錄傳回資料集
def creat_dataset_fromdir(directory,batch_size,isTrain = True):
    filenames, labels = list_images(directory)
    num_classes = len(set(labels))
    print("num_classes",num_classes)
    dataset = creat_batched_dataset(filenames, labels,batch_size,isTrain)
    return dataset,num_classes