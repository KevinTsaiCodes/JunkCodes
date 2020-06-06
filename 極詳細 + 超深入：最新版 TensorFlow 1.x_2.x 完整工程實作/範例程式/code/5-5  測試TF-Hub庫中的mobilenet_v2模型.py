# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


with open('中文標簽.csv','r+') as f: 		#開啟檔案
    labels =list( map(lambda x:x.replace(',',' '),list(f))  )
    print(len(labels),type(labels),labels[:5]) #顯示輸出中文標簽

sample_images = ['hy.jpg', 'ps.jpg','72.jpg']               #定義待測試圖片路徑

#載入分類別模型
module_spec = hub.load_module_spec("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2")

#module_spec = hub.load_module_spec("mobilenet_v2_100_224")

#module_spec = hub.load_module_spec(r"C:\Users\ljh\AppData\Local\Temp\tfhub_modules\bb6444e8248f8c581b7a320d5ff53061e4506c19")
height, width = hub.get_expected_image_size(module_spec)#獲得模型的輸入圖片尺寸

input_imgs = tf.placeholder(tf.float32, [None, height,width,3]) #定義占位符[batch_size, height, width, 3].
images = 2 *( input_imgs / 255.0)-1.0                         #歸一化圖片


module = hub.Module(module_spec)

logits = module(images)   # 輸出的形狀為 [batch_size, num_classes].
#也可以使用如下程式碼（以簽名的模式）
#  outputs = module(dict(images=images), signature="image_classification", as_dict=True)
#  logits = outputs["default"]

y = tf.argmax(logits,axis = 1)                          #獲得結果的輸出節點
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    def preimg(img):                                    #定義圖片預先處理函數
        return np.asarray(img.resize((height, width)),
                          dtype=np.float32).reshape(height, width,3)

    #獲得原始圖片與預先處理圖片
    batchImg = [ preimg( Image.open(imgfilename) ) for imgfilename in sample_images ]
    orgImg = [  Image.open(imgfilename)  for imgfilename in sample_images ]

    yv,img_norm = sess.run([y,images], feed_dict={input_imgs: batchImg})    #輸入到模型

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





