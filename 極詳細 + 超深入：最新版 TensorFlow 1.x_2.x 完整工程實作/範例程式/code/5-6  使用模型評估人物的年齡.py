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
from sklearn.utils import shuffle
import os

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


#載入標簽
data_dir = 'IMBD-WIKI\\'  #定義檔案路徑
_,labels = load_sample(data_dir,False)   #載入檔名稱與標簽
print(labels)#輸出load_sample傳回的標簽字串


sample_images = ['22.jpg', 'tt2t.jpg']               #定義待測試圖片路徑



tf.logging.set_verbosity(tf.logging.ERROR)
tf.reset_default_graph()
#分類別模型
thissavedir= 'tmp'
PATH_TO_CKPT = thissavedir +'/output_graph.pb'
od_graph_def = tf.GraphDef()
with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
#    print(od_graph_def)
    tf.import_graph_def(od_graph_def, name='')

fenlei_graph = tf.get_default_graph()


#print(fenlei_graph.get_operations())

height,width = 224,224

with tf.Session(graph=fenlei_graph) as sess:
    result = fenlei_graph.get_tensor_by_name('final_result:0')
    input_imgs = fenlei_graph.get_tensor_by_name('Placeholder:0')
    y = tf.argmax(result,axis = 1)


    def preimg(img):                                    #定義圖片預先處理函數
        reimg = np.asarray(img.resize((height, width)),
                          dtype=np.float32).reshape(height, width,3)
        normimg = 2 *( reimg / 255.0)-1.0
        return normimg

    #獲得原始圖片與預先處理圖片
    batchImg = [ preimg( Image.open(imgfilename) ) for imgfilename in sample_images ]
    orgImg = [  Image.open(imgfilename)  for imgfilename in sample_images ]

    yv = sess.run(y, feed_dict={input_imgs: batchImg})
    print(yv)

    print(yv,np.shape(yv))                                  #顯示輸出結果
    def showresult(yy,img_norm,img_org):                    #定義顯示圖片函數
        plt.figure()
        p1 = plt.subplot(121)
        p2 = plt.subplot(122)
        p1.imshow(img_org)# 顯示圖片
        p1.axis('off')
        p1.set_title("organization image")

        img = ((img_norm+1)/2)*255
        p2.imshow(  np.asarray(img,np.uint8)      )# 顯示圖片
        p2.axis('off')
        p2.set_title("input image")

        plt.show()

        print("索引：",yy,",","年紀：",labels[yy])

    for yy,img1,img2 in zip(yv,batchImg,orgImg):            #顯示每條結果及圖片
        showresult(yy,img1,img2)




