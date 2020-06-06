# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室 
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

annFile='./cocos2014/annotations_trainval2014/annotations/instances_train2014.json'
coco=COCO(annFile)#載入註釋的json資料

cats = coco.loadCats(coco.getCatIds())#分析分類別訊息
print(cats,len(cats))#80個分類別
nmcats=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nmcats)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))
print("supercategory len",len(nms))#12個超級分類別

# 分類別並不連續 ，例如：沒有26.第一個是1，  最後一個是90.
catIds = coco.getCatIds(catNms=nmcats)
print(catIds)  

#根據類別名獲得對應的圖片清單
catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds )  
print(catIds,len(imgIds),imgIds[:5])

#從指定清單中取一張圖片
index = imgIds[np.random.randint(0,len(imgIds))]
print(index)
img = coco.loadImgs(index)[0]#index可以是陣列。會傳回多個圖片
print(img)
I = io.imread(img['coco_url'])
plt.axis('off')
plt.imshow(I)
plt.show()


#獲得標注的分割訊息，並疊加到原圖顯示出來
plt.imshow(I); plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)#iscrowd代表是否是一群
anns = coco.loadAnns(annIds)#一條標注ID對應的訊息（segmentation（分割）、bbox（框）、category_id（類別別））
print(annIds,anns)
coco.showAnns(anns)#將分割的訊息疊加到圖形上

#載入關鍵點json
annFile = './annotations_trainval2014/annotations/person_keypoints_train2014.json'
coco_kps=COCO(annFile)
plt.imshow(I); plt.axis('off')
ax = plt.gca()
annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco_kps.loadAnns(annIds)#超級類別person的每條標注，內含了關鍵點 及segmentation和bbox、category_id
print(annIds,anns)
coco_kps.showAnns(anns)

#載入圖片描述json
annFile = './annotations_trainval2014/annotations/captions_train2014.json'
coco_caps=COCO(annFile)
annIds = coco_caps.getAnnIds(imgIds=img['id']);#每一個圖片id,對應多條描述
anns = coco_caps.loadAnns(annIds)#跟據描述id，載入每條描述
print(annIds,anns)#每條描述內含id 圖片id 和一句話
coco_caps.showAnns(anns)


