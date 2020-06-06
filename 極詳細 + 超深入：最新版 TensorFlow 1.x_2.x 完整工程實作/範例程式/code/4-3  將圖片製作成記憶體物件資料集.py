"""
@author: 程式碼醫生工作室
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""

import tensorflow as tf
import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.utils import shuffle

def load_sample(sample_dir):
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
    return shuffle(np.asarray( lfilenames),np.asarray( labels)),np.asarray(lab)


data_dir = 'mnist_digits_images\\'  #定義檔案路徑

(image,label),labelsnames = load_sample(data_dir)   #載入檔名稱與標簽
print(len(image),image[:2],len(label),label[:2])#輸出load_sample傳回的資料結果
print(labelsnames[ label[:2] ],labelsnames)#輸出load_sample傳回的標簽字串


def get_batches(image,label,input_w,input_h,channels,batch_size):

    queue = tf.train.slice_input_producer([image,label])  #使用tf.train.slice_input_producer實現一個輸入的佇列
    label = queue[1]                                        #從輸加入佇列列裡讀取標簽

    image_c = tf.read_file(queue[0])                        #從輸加入佇列列裡讀取image路徑

    image = tf.image.decode_bmp(image_c,channels)           #按照路徑讀取圖片

    image = tf.image.resize_image_with_crop_or_pad(image,input_w,input_h) #修改圖片大小


    image = tf.image.per_image_standardization(image) #圖形標准化處理，(x - mean) / adjusted_stddev

    image_batch,label_batch = tf.train.batch([image,label],#呼叫tf.train.batch函數產生批次資料
               batch_size = batch_size,
               num_threads = 64)

    images_batch = tf.cast(image_batch,tf.float32)   #將資料型態轉為float32

    labels_batch = tf.reshape(label_batch,[batch_size])#修改標簽的形狀shape
    return images_batch,labels_batch


batch_size = 16
image_batches,label_batches = get_batches(image,label,28,28,1,batch_size)



def showresult(subplot,title,thisimg):          #顯示單一圖片
    p =plt.subplot(subplot)
    p.axis('off')
    #p.imshow(np.asarray(thisimg[0], dtype='uint8'))
    p.imshow(np.reshape(thisimg, (28, 28)))
    p.set_title(title)

def showimg(index,label,img,ntop):   #顯示
    plt.figure(figsize=(20,10))     #定義顯示圖片的寬、高
    plt.axis('off')
    ntop = min(ntop,9)
    print(index)
    for i in range (ntop):
        showresult(100+10*ntop+1+i,label[i],img[i])
    plt.show()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)  #起始化

    coord = tf.train.Coordinator()          #開啟列隊
    threads = tf.train.start_queue_runners(sess = sess,coord = coord)
    try:
        for step in np.arange(10):
            if coord.should_stop():
                break
            images,label = sess.run([image_batches,label_batches]) #植入資料

            showimg(step,label,images,batch_size)       #顯示圖片
            print(label)                                 #列印資料

    except tf.errors.OutOfRangeError:
        print("Done!!!")
    finally:
        coord.request_stop()

    coord.join(threads)                             #關閉列隊

