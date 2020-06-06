# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室 
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""

import tensorflow as tf
model = __import__("5-2  model")
MyNASNetModel = model.MyNASNetModel

import sys                                      
nets_path = r'slim'                             #載入環境變數
if nets_path not in sys.path:
    sys.path.insert(0,nets_path)
else:
    print('already add slim')
    
from nets.nasnet import nasnet                 #匯出nasnet
slim = tf.contrib.slim                         #slim
image_size = nasnet.build_nasnet_mobile.default_image_size  #獲得圖片輸入尺寸 224

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

batch_size = 32
test_dir  = 'data/val'



def check_accuracy(sess):
    """
    測試模型準確率
    """
    sess.run(mymode.test_init_op)  #起始化測試資料集
    num_correct, num_samples = 0, 0 #定義正確個數 和 總個數
    i = 0
    while True:
        i+=1
        print('i',i)
        try:
            #計算correct_prediction 取得prediction、labels是否相同 
            correct_pred,accuracy,logits = sess.run([mymode.correct_prediction,mymode.accuracy,mymode.logits])
            #累加correct_pred
            num_correct += correct_pred.sum()
            num_samples += correct_pred.shape[0]
            print("accuracy",accuracy,logits)

        
        except tf.errors.OutOfRangeError:  #捕捉例外，資料用完自動跳出
            print('over')
            break
    
    acc = float(num_correct) / num_samples #計算並傳回準確率
    return acc 


def check_sex(imgdir,sess):
    img = Image.open(image_dir)                               #讀入圖片
    if "RGB"!=img.mode :                                      #檢查圖片格式
        img = img.convert("RGB") 

    img = np.asarray(img.resize((image_size,image_size)),     #圖形預先處理  
                          dtype=np.float32).reshape(1,image_size,image_size,3)
    img = 2 *( img / 255.0)-1.0 

#一批次資料    
#    tt = img
#    for nn in range(31):
#        tt= np.r_[tt,img]
#    print(np.shape(tt))
    
    prediction = sess.run(mymode.logits, {mymode.images: img})#傳入nasnet輸入端中
    print(prediction)
    
    pre = prediction.argmax()#傳回張量中最大值的索引

    print(pre)
    
    if pre == 1: img_id = 'man'
    elif pre == 2: img_id = 'woman'
    else: img_id = 'None'
    plt.imshow( np.asarray((img[0]+1)*255/2,np.uint8 )  )
    plt.show()
    print(img_id,"--",image_dir)#傳回類別別
    return pre
    

mymode = MyNASNetModel()                 #起始化模型
mymode.build_model('test',test_dir )     #將模型定義載入圖中

with tf.Session() as sess:  
    #載入模型
    mymode.load_cpk(mymode.global_step,sess,1,mymode.saver,mymode.save_path )

    #測試模型的準確性
    val_acc = check_accuracy(sess)
    print('Val accuracy: %f\n' % val_acc)

    #單張圖片測試
    image_dir = 'tt2t.jpg'         #選取測試圖片
    check_sex(image_dir,sess)
    
    image_dir = test_dir + '\\woman' + '\\000001.jpg'         #選取測試圖片
    check_sex(image_dir,sess)
    
    image_dir = test_dir + '\\man' + '\\000003.jpg'         #選取測試圖片
    check_sex(image_dir,sess)
