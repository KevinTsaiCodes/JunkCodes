# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""

import os
import tensorflow as tf

from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt

tf.enable_eager_execution()
print("TensorFlow 版本: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))




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
(filenames,labels),_ =load_sample(directory,shuffleflag=False) #載入檔名稱與標簽


def _distorted_image(image,size,ch=1,shuffleflag = False,cropflag  = False,
                     brightnessflag=False,contrastflag=False):    #定義函數，實現變化圖片
    distorted_image =tf.image.random_flip_left_right(image)

    if cropflag == True:                                                #隨機裁剪
        s = tf.random_uniform((1,2),int(size[0]*0.8),size[0],tf.int32)
        distorted_image = tf.random_crop(distorted_image, [s[0][0],s[0][0],ch])

    distorted_image = tf.image.random_flip_up_down(distorted_image)#上下隨機翻轉
    if brightnessflag == True:#隨機變化亮度
        distorted_image = tf.image.random_brightness(distorted_image,max_delta=10)
    if contrastflag == True:   #隨機變化比較度
        distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)
    if shuffleflag==True:
        distorted_image = tf.random_shuffle(distorted_image)#沿著第0維亂序
    return distorted_image


def _norm_image(image,size,ch=1,flattenflag = False):    #定義函數，實現歸一化，並且拍平
    image_decoded = image/255.0
    if flattenflag==True:
        image_decoded = tf.reshape(image_decoded, [size[0]*size[1]*ch])
    return image_decoded

from skimage import transform
def _random_rotated30(image, label): #定義函數實現圖片隨機旋轉動作

    def _rotated(image):                #封裝好的skimage模組，來進行圖片旋轉30度
        shift_y, shift_x = np.array(image.shape[:2],np.float32) / 2.
        tf_rotate = transform.SimilarityTransform(rotation=np.deg2rad(30))
        tf_shift = transform.SimilarityTransform(translation=[-shift_x, -shift_y])
        tf_shift_inv = transform.SimilarityTransform(translation=[shift_x, shift_y])
        image_rotated = transform.warp(image, (tf_shift + (tf_rotate + tf_shift_inv)).inverse)
        return image_rotated

    def _rotatedwrap():
        image_rotated = tf.py_function( _rotated,[image],[tf.float64])   #呼叫第三方函數
        return tf.cast(image_rotated,tf.float32)[0]

    a = tf.random_uniform([1],0,2,tf.int32)#實現隨機功能
    image_decoded = tf.cond(tf.equal(tf.constant(0),a[0]),lambda: image,_rotatedwrap)

    return image_decoded, label



def dataset(directory,size,batchsize,random_rotated=False):#定義函數，建立資料集
    """ parse  dataset."""
    (filenames,labels),_ =load_sample(directory,shuffleflag=False) #載入檔名稱與標簽
    def _parseone(filename, label):                         #解析一個圖片檔案
        """ Reading and handle  image"""
        image_string = tf.read_file(filename)         #讀取整個檔案
        image_decoded = tf.image.decode_image(image_string)
        image_decoded.set_shape([None, None, None])    # 必須有這句，不然下面會轉化失敗
        image_decoded = _distorted_image(image_decoded,size)#對圖片做扭曲變化
        image_decoded = tf.image.resize(image_decoded, size)  #變化尺寸
        image_decoded = _norm_image(image_decoded,size)#歸一化
        image_decoded = tf.cast(image_decoded,dtype=tf.float32)
        label = tf.cast(  tf.reshape(label, []) ,dtype=tf.int32  )#將label 轉為張量
        return image_decoded, label

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))#產生Dataset物件
    dataset = dataset.map(_parseone)   #有圖片內容的資料集

    if random_rotated == True:
        dataset = dataset.map(_random_rotated30)

    dataset = dataset.batch(batchsize) #批次劃分資料集

    return dataset




#若果顯示有錯，可以嘗試使用np.reshape(thisimg, (size[0],size[1],3))或
#np.asarray(thisimg[0], dtype='uint8')改變型態與形狀
def showresult(subplot,title,thisimg):          #顯示單一圖片
    p =plt.subplot(subplot)
    p.axis('off')
    p.imshow(thisimg)
    p.set_title(title)

def showimg(index,label,img,ntop):   #顯示
    plt.figure(figsize=(20,10))     #定義顯示圖片的寬、高
    plt.axis('off')
    ntop = min(ntop,9)
    print(index)
    for i in range (ntop):
        showresult(100+10*ntop+1+i,label[i],img[i])
    plt.show()

def getone(dataset):
    iterator = dataset.make_one_shot_iterator()			#產生一個迭代器
    one_element = iterator.get_next()					#從iterator裡取出一個元素
    return one_element

sample_dir=r"man_woman"
size = [96,96]
batchsize = 10
tdataset = dataset(sample_dir,size,batchsize)
tdataset2 = dataset(sample_dir,size,batchsize,True)
print(tdataset.output_types)  #列印資料集的輸出訊息
print(tdataset.output_shapes)

for step,value in enumerate(tdataset):
    showimg(step, value[1].numpy(),np.asarray( value[0]*255,np.uint8),10)       #顯示圖片


