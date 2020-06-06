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
from pycocotools.coco import COCO
import skimage.io as io

mask_rcnn_model = __import__("8-24  mask_rcnn_model")
MaskRCNN = mask_rcnn_model.MaskRCNN
utils = __import__("8-25  mask_rcnn_utils")
visualize = __import__("8-26  mask_rcnn_visualize")

################################載入資料集
annFile='./cocos2014/annotations_trainval2014/annotations/instances_train2014.json'
coco=COCO(annFile)#載入註釋的json資料

class_ids = sorted(coco.getCatIds())#獲得分類別id
class_info = coco.loadCats(coco.getCatIds())#分析分類別訊息
class_name=[n["name"] for n in class_info]

class_ids.insert(0,0)
class_name.insert(0,"BG")

print(class_ids)#所有的類別索引
print(class_name)#所有的類別名


#################################################載入模型

BATCH_SIZE =1#批次

MODEL_DIR = "./log"
#可以讓gpu進行訓練，cpu進行檢驗。但是實際中，若果gpu組態低，直接gpu有可能會核心死掉。
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
# Create model in inference mode
with tf.device(DEVICE):
    model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, num_class=len(class_ids),batch_size = BATCH_SIZE)#加完背景後81個類別

#模型權重檔案路徑
weights_path = "./mask_rcnn_coco.h5"

#載入權重檔案
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

# Show stats of all trainable weights    
utils.html_weight_stats(model)#顯示權重
######################################################

################################################取得一個檔案
#根據類別名獲得對應的圖片清單
catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds )  
print(catIds,len(imgIds),imgIds[:5])

#從指定清單中取一張圖片
index = imgIds[np.random.randint(0,len(imgIds))]
print(index)
img = coco.loadImgs(index)[0]#index可以是陣列。會傳回多個圖片
print(img)
image = io.imread(img['coco_url'])#從網路載入一個圖片
plt.axis('off')
plt.imshow(image)
plt.show()

####################################################################
#骨干網結果
  
ResNetFeatures = utils.run_graph(model,[image], [
    ("res4w_out",          model.keras_model.get_layer("res4w_out").output),  # for resnet100
    ("res5c_out",          model.keras_model.get_layer("res5c_out").output),  # for resnet100
],BATCH_SIZE)
    
# Backbone feature map(1, 64, 64, 1024)
visualize.display_images(np.transpose(ResNetFeatures["res4w_out"][0,:,:,:4], [2, 0, 1]))
visualize.display_images(np.transpose(ResNetFeatures["res5c_out"][0,:,:,:4], [2, 0, 1]))

##################################################
#FPN結果
roi_align_mask = utils.run_graph(model,[image], [
    ("fpn_p2",          model.keras_model.get_layer("fpn_p2").output),  #
    ("fpn_p3",          model.keras_model.get_layer("fpn_p3").output),  #
    ("fpn_p4",          model.keras_model.get_layer("fpn_p4").output),  #
    ("fpn_p5",          model.keras_model.get_layer("fpn_p5").output),  #   
    ("fpn_p6",          model.keras_model.get_layer("fpn_p6").output),  #    
],BATCH_SIZE)


###################################################
#第一步 RPN網路
    
# Run RPN sub-graph
pillar = model.keras_model.get_layer("ROI").output  #獲得ROI節點，即 ProposalLayer層

rpn = utils.run_graph(model,[image], [
    ("rpn_class", model.keras_model.get_layer("rpn_class").output),#(1, 261888, 2) 
    ("pre_nms_anchors", model.ancestor(pillar, "ROI/pre_nms_anchors:0")),
    ("refined_anchors", model.ancestor(pillar, "ROI/refined_anchors:0")),
    ("refined_anchors_clipped", model.ancestor(pillar, "ROI/refined_anchors_clipped:0")),
    ("post_nms_anchor_ix", model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV3:0") ),#shape: (1000,)
    ("proposals", model.keras_model.get_layer("ROI").output),
],BATCH_SIZE)

print(rpn['rpn_class'][0,:3])#將rpn網路的前三個元素列印出來
print(rpn['pre_nms_anchors'][0,:3])#將rpn網路的前三個元素列印出來    
print(model.anchors[:3])#將rpn網路的前三個元素列印出來    
    
def get_ax(rows=1, cols=1, size=16):#設定顯示的圖片位置及大小
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax
#將前50個分值高的錨點示出來
limit = 50
h, w = mask_rcnn_model.IMAGE_DIM,mask_rcnn_model.IMAGE_DIM;
pre_nms_anchors = rpn['pre_nms_anchors'][0, :limit] * np.array([h, w, h, w])
print(image.shape)
image2, window, scale, padding, _ = utils.resize_image( image, 
                                    min_dim=mask_rcnn_model.IMAGE_MIN_DIM, 
                                    max_dim=mask_rcnn_model.IMAGE_MAX_DIM,
                                    mode=mask_rcnn_model.IMAGE_RESIZE_MODE)
print(image2.shape)
visualize.draw_boxes(image2, boxes=pre_nms_anchors, ax=get_ax())


#rpn_class (1, 261888, 2)  背景和前景。softmax後的值。1代表正值， 由大到小的
sorted_anchor_ids = np.argsort(rpn['rpn_class'][:,:,1].flatten())[::-1]
visualize.draw_boxes(image2, boxes=model.anchors[sorted_anchor_ids[:limit]], ax=get_ax())



#隨意取50個帶nms之前的資料，和調整後的。  還有規整完調整後的  (1, 6000, 4)
ax = get_ax(1, 2)
pre_nms_anchors = rpn['pre_nms_anchors'][0, :limit] * np.array([h, w, h, w])
refined_anchors = rpn['refined_anchors'][0, :limit] * np.array([h, w, h, w])
refined_anchors_clipped = rpn['refined_anchors_clipped'][0, :limit] * np.array([h, w, h, w])
#取50個在nms之前的資料，邊框調整後和邊框剪輯後的
visualize.draw_boxes(image2, boxes=pre_nms_anchors,refined_boxes=refined_anchors, ax=ax[0])
visualize.draw_boxes(image2, refined_boxes=refined_anchors_clipped, ax=ax[1])#邊框剪輯後的
######################################################

#用nms之後(1000,)為實際的索引值
post_nms_anchor_ix = rpn['post_nms_anchor_ix'][ :limit]
refined_anchors_clipped = rpn["refined_anchors_clipped"][0, post_nms_anchor_ix] * np.array([h, w, h, w])
visualize.draw_boxes(image2, refined_boxes=refined_anchors_clipped, ax=get_ax())


# Convert back to image coordinates for display(1, 1000, 4) 
proposals = rpn['proposals'][0, :limit] * np.array([h, w, h, w])
visualize.draw_boxes(image2, refined_boxes=proposals, ax=get_ax())

########################################################

######################################################
#第二步
    
roi_align_classifierlar = model.keras_model.get_layer("roi_align_classifier").output  #獲得ROI節點，即 ProposalLayer層

roi_align_classifier = utils.run_graph(model,[image], [
    ("roi_align_classifierlar", model.keras_model.get_layer("roi_align_classifier").output),#(1, 261888, 2) 
    ("ix", model.ancestor(roi_align_classifierlar, "roi_align_classifier/ix:0")),
    ("level_boxes", model.ancestor(roi_align_classifierlar, "roi_align_classifier/level_boxes:0")),
    ("box_indices", model.ancestor(roi_align_classifierlar, "roi_align_classifier/Cast_2:0")),

 ],BATCH_SIZE)
    
print(roi_align_classifier["ix"][:5])    #(828, 2) 
print(roi_align_classifier["level_boxes"][:5])  #(828, 4)      
print(roi_align_classifier["box_indices"][:5])  #(828, 4) 



#分類別器結果
fpn_classifier = utils.run_graph(model,[image], [
    ("probs", model.keras_model.get_layer("mrcnn_class").output),#shape: (1, 1000, 81)
    ("deltas", model.keras_model.get_layer("mrcnn_bbox").output),#(1, 1000, 81, 4) 
],BATCH_SIZE)
#相對於reshape後的框，所以要使用image2
proposals=utils.denorm_boxes(rpn["proposals"][0], image2.shape[:2])#(1000, 4)

#81類別中的最大索引--代表class id(索引就是分類別)
roi_class_ids = np.argmax(fpn_classifier["probs"][0], axis=1)#(1000,)
print(roi_class_ids.shape,roi_class_ids[:20])
roi_class_names = np.array(class_name)[roi_class_ids]#根據索引把名字取出來
print(roi_class_names[:20])
#去重，類別別個數
print(list(zip(*np.unique(roi_class_names, return_counts=True))))

roi_positive_ixs = np.where(roi_class_ids > 0)[0]#不是背景的類別索引
print("{}中有{}個前景案例\n{}".format(len(proposals),len(roi_positive_ixs),roi_positive_ixs))

#根據索引將最大的那個值取出來。當作分數
roi_scores = np.max(fpn_classifier["probs"][0],axis=1)
print(roi_scores.shape,roi_scores[:20])

############################
#邊框可視化
limit = 50
ax = get_ax(1, 2)

ixs = np.random.randint(0, proposals.shape[0], limit)
captions = ["{} {:.3f}".format(class_name[c], s) if c > 0 else ""
            for c, s in zip(roi_class_ids[ixs], roi_scores[ixs])]

visib= np.where(roi_class_ids[ixs] > 0, 2, 1)#前景統一設為2，背景設為1

visualize.draw_boxes(image2, boxes=proposals[ixs],  #原始的框放進去
                     visibilities=visib,#2突出顯示.1一般顯示
                     captions=captions, title="before fpn_classifier", ax=ax[0])

#把指定類別索引的座標分析出來
#取出每個框對應分類別的座標偏移。fpn_classifier["deltas"]形狀為(1, 1000, 81, 4)
roi_bbox_specific = fpn_classifier["deltas"][0, np.arange(proposals.shape[0]), roi_class_ids]
print("roi_bbox_specific", roi_bbox_specific)#( 1000,  4)

#根據偏移來調整ROI Shape: [N, (y1, x1, y2, x2)]
refined_proposals = utils.apply_box_deltas(
    proposals, roi_bbox_specific * mask_rcnn_model.BBOX_STD_DEV).astype(np.int32)
print("refined_proposals", refined_proposals)

limit =5
ids = np.random.randint(0, len(roi_positive_ixs), limit)  #取出5個不是背景的類別

captions = ["{} {:.3f}".format(class_name[c], s) if c > 0 else ""
            for c, s in zip(roi_class_ids[roi_positive_ixs][ids], roi_scores[roi_positive_ixs][ids])]

visualize.draw_boxes(image2, boxes=proposals[roi_positive_ixs][ids],
                     refined_boxes=refined_proposals[roi_positive_ixs][ids],
                     captions=captions, title="After fpn_classifier",ax=ax[1])


#若果將座標按照圖片image來變化，還需要如下的方法轉成image2的尺寸
#proposals=utils.denorm_boxes(rpn["proposals"][0], image.shape[:2])#(1000, 4)
#captions = ["{} {:.3f}".format(class_name[c], s) if c > 0 else ""
#            for c, s in zip(roi_class_ids[roi_positive_ixs], roi_scores[roi_positive_ixs])]
#rpnbox = rpn["proposals"][0]
#
#print(proposals[roi_positive_ixs][:5])
#coord_norm = utils.norm_boxes(proposals[roi_positive_ixs],image2.shape[:2])
#window_norm = utils.norm_boxes(window, image2.shape[:2])
#print(window)
#print(window_norm)
#coorded_norm = refineboxbywindow(window_norm,rpnbox)
#bbbox = utils.denorm_boxes(coorded_norm, image.shape[:2])[roi_positive_ixs]
#print(bbbox)
#
#visualize.draw_boxes(image, boxes=proposals[roi_positive_ixs],
#                     refined_boxes= bbbox,
#                     captions=captions, title="ROIs After Refinement",ax=ax[1])


#########################################################
#實物邊框檢驗

#按照視窗縮放,來調整座標
def refineboxbywindow(window,coordinates):

    wy1, wx1, wy2, wx2 = window
    shift = np.array([wy1, wx1, wy1, wx1])
    wh = wy2 - wy1  # window height
    ww = wx2 - wx1  # window width
    scale = np.array([wh, ww, wh, ww])
    #按照視窗縮放座標
    refine_coordinates = np.divide(coordinates - shift, scale)
    return refine_coordinates

#模型輸出的最終檢驗目的結果
DetectionLayer = utils.run_graph(model,[image], [
        #(1, 100, 6) 6: 4個位置1個分類別1個分數
    ("detections", model.keras_model.get_layer("mrcnn_detection").output),
],BATCH_SIZE)



##獲得分類別的ID
det_class_ids = DetectionLayer['detections'][0, :, 4].astype(np.int32)

det_ids = np.where(det_class_ids != 0)[0]#取出前景類別不等於0的索引，
det_class_ids = det_class_ids[det_ids]#預測的分類別ID
#將分類別ID顯示出來
print("{} detections: {}".format( len(det_ids), np.array(class_name)[det_class_ids]))

roi_scores= DetectionLayer['detections'][0, :, -1]#獲得分類別分數
print(roi_scores)
print(roi_scores[det_ids])

boxes_norm= DetectionLayer['detections'][0, :, :4]#
window_norm = utils.norm_boxes(window, image2.shape[:2])
boxes = refineboxbywindow(window_norm,boxes_norm)#按照視窗縮放,來調整座標

#將座標轉化為像素座標
refined_proposals=utils.denorm_boxes(boxes[det_ids], image.shape[:2])#(1000, 4)
captions = ["{} {:.3f}".format(class_name[c], s) if c > 0 else ""
            for c, s in zip(det_class_ids, roi_scores[det_ids])]

visualize.draw_boxes( image, boxes=refined_proposals[det_ids],
    visibilities=[2] * len(det_ids),#統一設為2，表示用實線顯示 
    captions=captions, title="Detections after NMS", ax=get_ax())

print(det_ids,refined_proposals)
print(det_class_ids)
#########################################################
#語義分割
#第三部 語義部分

#模型輸出的最終檢驗目的結果
maskLayer = utils.run_graph(model,[image], [
    ("masks", model.keras_model.get_layer("mrcnn_mask").output),#(1, 100, 28, 28, 81)
],BATCH_SIZE)

#按照特殊的類別索引，取出遮罩---該遮罩是每個框裡的相對位移[n,28,28]
det_mask_specific = np.array([maskLayer["masks"][0, i, :, :, c] 
                              for i, c in enumerate(det_class_ids)])
print(det_mask_specific.shape)

#復原成真實大小，按照圖片的框的位置來復原真實座標(n, image.h, image.h)
true_masks = np.array([utils.unmold_mask(m, refined_proposals[i], image.shape)
                      for i, m in enumerate(det_mask_specific)])

#遮罩可視化
visualize.display_images(det_mask_specific[:4] * 255, cmap="Blues", interpolation="none")
visualize.display_images(true_masks[:4] * 255, cmap="Blues", interpolation="none")

#語義分割結果可視化
t = np.transpose(true_masks,(1,2,0))
visualize.display_instances(image, refined_proposals, t, det_class_ids, 
                            class_name, roi_scores[det_ids])

##########################################################
#最終結果

results = model.detect([image], verbose=1)
#
# Visualize results
r = results[0]

#print("image", image)
#print("mask", r['masks'])
print("class_ids", r['class_ids'])
print("bbox", r['rois'])
#print("class_names", class_name)
print("scores", r['scores'])


visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_name, r['scores'])





