# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""

import sys                                         #起始化環境變數
nets_path = r'./slim'
if nets_path not in sys.path:
    sys.path.insert(0,nets_path)
else:
    print('already add slim')

import tensorflow as tf                           #引入頭檔案
from PIL import Image
from matplotlib import pyplot as plt
from nets.nasnet import pnasnet
import numpy as np
from datasets import imagenet
slim = tf.contrib.slim



tf.reset_default_graph()

image_size = pnasnet.build_pnasnet_large.default_image_size   #獲得圖片輸入尺寸
labels = imagenet.create_readable_names_for_imagenet_labels() #獲得資料集標簽
print(len(labels),labels)                                     #顯示輸出標簽

def getone(onestr):
    return onestr.replace(',',' ')

with open('中文標簽.csv','r+') as f: 		#開啟檔案
    labels =list( map(getone,list(f))  )
    print(len(labels),type(labels),labels[:5]) #顯示輸出中文標簽




sample_images = ['hy.jpg', 'ps.jpg','72.jpg']               #定義待測試圖片路徑

input_imgs = tf.placeholder(tf.float32, [None, image_size,image_size,3]) #定義占位符

x1 = 2 *( input_imgs / 255.0)-1.0                         #歸一化圖片

arg_scope = pnasnet.pnasnet_large_arg_scope()              #獲得模型命名空間
with slim.arg_scope(arg_scope):
    logits, end_points = pnasnet.build_pnasnet_large(x1,num_classes = 1001, is_training=False)
    prob = end_points['Predictions']
    y = tf.argmax(prob,axis = 1)                          #獲得結果的輸出節點


checkpoint_file = r'./pnasnet-5_large_2017_12_13/model.ckpt'   #定義模型路徑
saver = tf.train.Saver()                                #定義saver，用於載入模型
with tf.Session() as sess:                              #建立階段
    saver.restore(sess, checkpoint_file)                #載入模型

    def preimg(img):                                    #定義圖片預先處理函數
        ch = 3
        if img.mode=='RGBA':                            #相容RGBA圖片
            ch = 4

        imgnp = np.asarray(img.resize((image_size,image_size)),
                          dtype=np.float32).reshape(image_size,image_size,ch)
        return imgnp[:,:,:3]
    #獲得原始圖片與預先處理圖片
    batchImg = [ preimg( Image.open(imgfilename) ) for imgfilename in sample_images ]
    orgImg = [  Image.open(imgfilename)  for imgfilename in sample_images ]

    yv,img_norm = sess.run([y,x1], feed_dict={input_imgs: batchImg})    #輸入到模型

    print(yv,np.shape(yv))                                  #顯示輸出結果
    def showresult(yy,img_norm,img_org):                    #定義顯示圖片函數
        plt.figure()
        p1 = plt.subplot(121)
        p2 = plt.subplot(122)
        p1.imshow(img_org)# 顯示圖片
        p1.axis('off')
        p1.set_title("organization image")

        p2.imshow((img_norm * 255).astype(np.uint8))# 顯示圖片
        p2.axis('off')
        p2.set_title("input image")

        plt.show()

        print(yy,labels[yy])

    for yy,img1,img2 in zip(yv,batchImg,orgImg):            #顯示每條結果及圖片
        showresult(yy,img1,img2)

