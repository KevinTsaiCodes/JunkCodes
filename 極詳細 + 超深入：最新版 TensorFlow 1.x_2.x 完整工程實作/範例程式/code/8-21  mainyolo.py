# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室 
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""

import os
import tensorflow as tf
import glob
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import tensorflow.contrib.eager as tfe
generator = __import__("8-14  generator")
BatchGenerator = generator.BatchGenerator
box = __import__("8-15  box")
draw_boxes = box.draw_boxes
yolov3 = __import__("8-18  yolov3")
Yolonet = yolov3.Yolonet
yololoss = __import__("8-20  yololoss")
loss_fn = yololoss.loss_fn

tf.enable_eager_execution()

PROJECT_ROOT = os.path.dirname(__file__)#取得目前目錄
print(PROJECT_ROOT)

#定義coco錨點候選框
COCO_ANCHORS = [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]
#定義預訓練模型路徑
YOLOV3_WEIGHTS = os.path.join(PROJECT_ROOT, "yolov3.weights")
#定義分類別
LABELS = ['0',"1", "2", "3",'4','5','6','7','8', "9"]

#定義樣本路徑
ann_dir = os.path.join(PROJECT_ROOT,  "data", "ann", "*.xml")
img_dir = os.path.join(PROJECT_ROOT,  "data", "img")

train_ann_fnames = glob.glob(ann_dir)#取得該路徑下的xml檔案
   
imgsize =416
batch_size =2
#製作資料集
generator = BatchGenerator(train_ann_fnames,img_dir,
                           net_size=imgsize,
                           anchors=COCO_ANCHORS,
                             batch_size=2,
                             labels=LABELS,
                             jitter = False)#隨機變化尺寸。資料增強

#定義訓練參數
learning_rate = 1e-4  #定義研讀率
num_epoches =85       #定義迭代次數
save_dir = "./model"  #定義模型路徑

#循環整個資料集，進行loss值驗證
def _loop_validation(model, generator):
    n_steps = generator.steps_per_epoch
    loss_value = 0
    for _ in range(n_steps): #按批次循環取得資料，並計算loss
        xs, yolo_1, yolo_2, yolo_3 = generator.next_batch()
        xs=tf.convert_to_tensor(xs)
        yolo_1=tf.convert_to_tensor(yolo_1)
        yolo_2=tf.convert_to_tensor(yolo_2)
        yolo_3=tf.convert_to_tensor(yolo_3)        
        ys = [yolo_1, yolo_2, yolo_3]
        ys_ = model(xs )
        loss_value += loss_fn(ys, ys_,anchors=COCO_ANCHORS,
            image_size=[imgsize, imgsize] )
    loss_value /= generator.steps_per_epoch
    return loss_value

#循環整個資料集，進行模型訓練
def _loop_train(model,optimizer, generator,grad):
    # one epoch
    n_steps = generator.steps_per_epoch
    for _ in tqdm(range(n_steps)):#按批次循環取得資料，並計算訓練
        xs, yolo_1, yolo_2, yolo_3 = generator.next_batch()
        xs=tf.convert_to_tensor(xs)
        yolo_1=tf.convert_to_tensor(yolo_1)
        yolo_2=tf.convert_to_tensor(yolo_2)
        yolo_3=tf.convert_to_tensor(yolo_3)
        ys = [yolo_1, yolo_2, yolo_3]
        optimizer.apply_gradients(grad(model,xs, ys))

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_fname = os.path.join(save_dir, "weights")

yolo_v3 = Yolonet(n_classes=len(LABELS))#案例化yolo模型類別物件
yolo_v3.load_darknet_params(YOLOV3_WEIGHTS, skip_detect_layer=True)#載入預訓練模型

#定義改善器
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

#定義函數計算loss
def _grad_fn(yolo_v3, images_tensor, list_y_trues):
    logits = yolo_v3(images_tensor)   
    loss = loss_fn(list_y_trues, logits,anchors=COCO_ANCHORS,
            image_size=[imgsize, imgsize])
    return loss

grad = tfe.implicit_gradients(_grad_fn)#獲得計算梯度的函數

history = []
for i in range(num_epoches):
    _loop_train( yolo_v3,optimizer, generator,grad)#訓練

    loss_value = _loop_validation(yolo_v3, generator)#驗證
    print("{}-th loss = {}".format(i, loss_value))

    #收集loss
    history.append(loss_value)
    if loss_value == min(history):#只有loss創新低時再儲存模型
        print("    update weight {}".format(loss_value))
        yolo_v3.save_weights("{}.h5".format(save_fname))
################################################################
#使用模型

IMAGE_FOLDER = os.path.join(PROJECT_ROOT,  "data", "test","*.png")
img_fnames = glob.glob(IMAGE_FOLDER)

imgs = []   #存放圖片
for fname in img_fnames:#讀取圖片
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgs.append(img)

yolo_v3.load_weights(save_fname+".h5")#將訓練好的模型載入
import numpy as np
for img in imgs:  #依次傳入模型
    boxes, labels, probs = yolo_v3.detect(img, COCO_ANCHORS,imgsize)
    print(boxes, labels, probs)
    image = draw_boxes(img, boxes, labels, probs, class_labels=LABELS, desired_size=400)
    image = np.asarray(image,dtype= np.uint8)
    plt.imshow(image)
    plt.show()



