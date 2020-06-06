# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室 
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

usecoco = 1  #案例的示範模式。設為1，表示使用coco資料集

def convert_coco_bbox(size, box):
    """
    Introduction
    ------------
        計算box的長寬和原始圖形的長寬比值
    Parameters
    ----------
        size: 原始圖形大小
        box: 標注box的訊息
    Returns
        x, y, w, h 標注box和原始圖形的比值
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0 - 1
    y = (box[1] + box[3]) / 2.0 - 1
    w = box[2]
    h = box[3]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h

def load_cocoDataset(annfile):
    """
    Introduction
    ------------
        讀取coco資料集的標注訊息
    Parameters
    ----------
        datasets: 資料集名字清單
    """
    from pycocotools.coco import COCO
    #from six.moves import xrange
    data = []
    coco = COCO(annfile)
    cats = coco.loadCats(coco.getCatIds())
    coco.loadImgs()
    base_classes = {cat['id'] : cat['name'] for cat in cats}
    imgId_catIds = [coco.getImgIds(catIds = cat_ids) for cat_ids in base_classes.keys()]
    image_ids = [img_id for img_cat_id in imgId_catIds for img_id in img_cat_id ]
    for image_id in image_ids:
        annIds = coco.getAnnIds(imgIds = image_id)
        anns = coco.loadAnns(annIds)
        img = coco.loadImgs(image_id)[0]
        image_width = img['width']
        image_height = img['height']

        for ann in anns:
            box = ann['bbox']
            bb = convert_coco_bbox((image_width, image_height), box)
            data.append(bb[2:])
    return np.array(data)




if usecoco == 1:
    dataFile = r"E:\Mask_RCNN-master\cocos2014\annotations\instances_train2014.json"
    points = load_cocoDataset(dataFile)
else: 
    num_points = 100
    dimensions = 2
    points = np.random.uniform(0, 1000, [num_points, dimensions])


num_clusters = 5
config=tf.estimator.RunConfig(model_dir='./kmeansmodel',save_checkpoints_steps=100)

kmeans = tf.contrib.factorization.KMeansClustering(config= config,
    num_clusters=num_clusters, use_mini_batch=False,relative_tolerance=0.01)

#訓練部分
def input_fn():
  return tf.train.limit_epochs(
      tf.convert_to_tensor(points, dtype=tf.float32), num_epochs=300)
kmeans.train(input_fn)
print("訓練結束，score(cost) = {}".format(kmeans.score(input_fn)))

anchors = kmeans.cluster_centers()

box_w = points[:1000, 0]
box_h = points[:1000, 1]
#plt.scatter(box_h, box_w, c = 'r')
#
#print(len(anchors))
#anchors = np.asarray(anchors)
#print((anchors[:,0]))
#plt.scatter(anchors[:,0], anchors[:, 1], c = 'b')
#plt.show() 

#聚類別結果
def show_input_fn():
  return tf.train.limit_epochs(
      tf.convert_to_tensor(points[:1000], dtype=tf.float32), num_epochs=1)
cluster_indices =list( kmeans.predict_cluster_index(show_input_fn) )


plt.scatter(box_h, box_w, c=cluster_indices)
plt.colorbar()
plt.scatter(anchors[:,0], anchors[:, 1], s=800,c='r',marker='x')
plt.show()

if usecoco == 1:
    trueanchors = []
    for cluster in anchors:
        trueanchors.append([round(cluster[0] * 416), round(cluster[1] * 416)])
    print("在416*416上面，所聚類別的錨點候選框為：",trueanchors)
 

distance = list(kmeans.transform(show_input_fn))#獲得每個座標離中心點的距離  
predict = list(kmeans.predict(show_input_fn) )#對每個點進行預測
print(distance[0],predict[0]) #顯示內容

#取出第一個類別的資料。並按照中心點遠近排序
firstclassdistance= np.array([  p['all_distances'][0]  for p in predict if p['cluster_index']==0 ])
dataindexsort= np.argsort(firstclassdistance)
print(len(dataindexsort),dataindexsort[:10],firstclassdistance[dataindexsort[:10]])


  
  
  
  
  
  