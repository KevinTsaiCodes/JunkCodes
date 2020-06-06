"""
@author: 程式碼醫生工作室
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""

import os
import tensorflow as tf
from PIL import Image
from sklearn.utils import shuffle
import numpy as np
from tqdm import tqdm

def load_sample(sample_dir,shuffleflag = True):
    '''遞歸讀取檔案。只支援一級。傳回檔名、數值標簽、數值對應的標簽名'''
    print ('loading sample  dataset..')
    lfilenames = []
    labelsnames = []
    for (dirpath, dirnames, filenames) in os.walk(sample_dir):#遞歸檢查資料夾
        for filename in filenames:                            #檢查所有檔名
            #print(dirnames)
            filename_path = os.sep.join([dirpath, filename])
            lfilenames.append(filename_path)               #加入檔名
            labelsnames.append( dirpath.split('\\')[-1] )#加入檔名對應的標簽

    lab= list(sorted(set(labelsnames)))  #產生標簽名稱清單
    labdict=dict( zip( lab  ,list(range(len(lab)))  )) #產生字典

    labels = [labdict[i] for i in labelsnames]
    if shuffleflag == True:
        return shuffle(np.asarray( lfilenames),np.asarray( labels)),np.asarray(lab)
    else:
        return (np.asarray( lfilenames),np.asarray( labels)),np.asarray(lab)



directory='man_woman\\'                                                     #定義樣本路徑
(filenames,labels),_ = load_sample(directory,shuffleflag=False)   #載入檔名稱與標簽


def makeTFRec(filenames,labels): #定義函數產生TFRecord
    writer= tf.python_io.TFRecordWriter("mydata.tfrecords") #透過tf.python_io.TFRecordWriter 寫入到TFRecords檔案
    for i in tqdm( range(0,len(labels) ) ):
        img=Image.open(filenames[i])
        img = img.resize((256, 256))
        img_raw=img.tobytes()#將圖片轉化為二進位格式
        example = tf.train.Example(features=tf.train.Features(feature={
                #存放圖片的標簽label
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[i]])),
                #存放實際的圖片
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            })) #example物件對label和image資料進行封裝

        writer.write(example.SerializeToString())  #序列化為字串
    writer.close()  #資料集製作完成

makeTFRec(filenames,labels)

################將tf資料集轉化為圖片##########################
def read_and_decode(filenames,flag = 'train',batch_size = 3):
    #根據檔名產生一個佇列
    if flag == 'train':
        filename_queue = tf.train.string_input_producer(filenames)#預設已經是shuffle並且循環讀取
    else:
        filename_queue = tf.train.string_input_producer(filenames,num_epochs = 1,shuffle = False)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #傳回檔名和檔案
    features = tf.parse_single_example(serialized_example, #取出包括image和label的feature物件
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    #tf.decode_raw可以將字串解析成圖形對應的像素陣列
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image, [256,256,3])
    #
    label = tf.cast(features['label'], tf.int32)

    if flag == 'train':
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5     #歸一化
        img_batch, label_batch = tf.train.batch([image, label],   #還可以使用tf.train.shuffle_batch進行亂序批次
                                                batch_size=batch_size, capacity=20)
#        img_batch, label_batch = tf.train.shuffle_batch([image, label],
#                                        batch_size=batch_size, capacity=20,
#                                        min_after_dequeue=10)
        return img_batch, label_batch

    return image, label

#############################################################
TFRecordfilenames = ["mydata.tfrecords"]
image, label =read_and_decode(TFRecordfilenames,flag='test')  #以測試的模式開啟資料集


saveimgpath = 'show\\'    #定義儲存圖片路徑
if tf.gfile.Exists(saveimgpath):  #若果存在saveimgpath，將其移除
    tf.gfile.DeleteRecursively(saveimgpath)  #也可以使用shutil.rmtree(saveimgpath)
tf.gfile.MakeDirs(saveimgpath)    #建立saveimgpath路徑

#開始一個階段讀取資料
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())   #起始化本機變數，沒有這句會顯示出錯
    #啟動多執行緒
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
    myset = set([])

    try:
        i = 0
        while True:
            example, examplelab = sess.run([image,label])#在階段中取出image和label
            examplelab = str(examplelab)
            if examplelab not in myset:
                myset.add(examplelab)
                tf.gfile.MakeDirs(saveimgpath+examplelab)
                print(saveimgpath+examplelab,i)
            img=Image.fromarray(example, 'RGB')#轉換Image格式
            img.save(saveimgpath+examplelab+'/'+str(i)+'_Label_'+'.jpg')#存下圖片
            print( i)
            i = i+1
    except tf.errors.OutOfRangeError:
        print('Done Test -- epoch limit reached')
    finally:
        coord.request_stop()
        coord.join(threads)
        print("stop()")
#############################################################
#訓練模式
image, label =read_and_decode(TFRecordfilenames)  #以訓練的模式開啟資料集
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())   #起始化本機變數，沒有這句會顯示出錯
    #啟動多執行緒
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
    myset = set([])
    try:
        for i in range(5):
            example, examplelab = sess.run([image,label])#在階段中取出image和label

            dirtrain = saveimgpath+"train_"+str(i)
            print(dirtrain,examplelab)
            tf.gfile.MakeDirs(dirtrain)
            for lab in range(len(examplelab)):
                print(lab)
                img=Image.fromarray(example[lab], 'RGB')#這裡Image是之前提到的
                img.save(dirtrain+'/'+str(lab)+'_Label_'+str(examplelab[lab])+'.jpg')#存下圖片

    except tf.errors.OutOfRangeError:
        print('Done Test -- epoch limit reached')
    finally:
        coord.request_stop()
        coord.join(threads)
        print("stop()")