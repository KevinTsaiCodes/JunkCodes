# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室 
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""

import math
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

mask_rcnn_model = __import__("8-29  mask_rcnn_model")
MaskRCNN = mask_rcnn_model.MaskRCNN
utils = __import__("8-30  mask_rcnn_utils")
visualize = __import__("8-32  mask_rcnn_visualize")

#隨機產生圖片類別
class ShapesDataset():

    def __init__(self, class_map=None):
        self.image_ids = []
        self.image_info = []
        #第0個類別是背景
        self.class_info = [{ "id": 0, "name": "BG"}]
        self.source_class_ids = {}
        
    def load_shapes(self, count, height, width):
        # Add classes
        self.class_info.append({ "id": 1, "name": "square"})
        self.class_info.append({ "id": 2, "name": "circle"})
        self.class_info.append({ "id": 3, "name": "triangle"})

        # Add images
        for i in range(count):
            bg_color, shapes = self.random_image(height, width)
            self.image_info.append({"width":width,"height":height,"bg_color":bg_color,"shapes":shapes})

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
        image = np.ones([info['height'], info['width'], 3], dtype=np.uint8)
        image = image * bg_color.astype(np.uint8)
        for shape, color, dims in info['shapes']:
            image = self.draw_shape(image, shape, dims, color)
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        shapes = info['shapes']
        count = len(shapes)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i, (shape, _, dims) in enumerate(info['shapes']):
            mask[:, :, i:i + 1] = self.draw_shape(mask[:, :, i:i + 1].copy(),
                                                  shape, dims, 1)
        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(
                occlusion, np.logical_not(mask[:, :, i]))
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        return mask, class_ids.astype(np.int32)

    def draw_shape(self, image, shape, dims, color):
        """Draws a shape from the given specs."""
        # Get the center x, y and the size s
        x, y, s = dims
        if shape == 'square':
            image = cv2.rectangle(image, (x - s, y - s),
                                  (x + s, y + s), color, -1)
        elif shape == "circle":
            image = cv2.circle(image, (x, y), s, color, -1)
        elif shape == "triangle":
            points = np.array([[(x, y - s),
                                (x - s / math.sin(math.radians(60)), y + s),
                                (x + s / math.sin(math.radians(60)), y + s),
                                ]], dtype=np.int32)
            image = cv2.fillPoly(image, points, color)
        return image

    def random_shape(self, height, width):
        """Generates specifications of a random shape that lies within
        the given height and width boundaries.
        Returns a tuple of three valus:
        * The shape name (square, circle, ...)
        * Shape color: a tuple of 3 values, RGB.
        * Shape dimensions: A tuple of values that define the shape size
                            and location. Differs per shape type.
        """
        # Shape
        shape = random.choice(["square", "circle", "triangle"])
        # Color
        color = tuple([random.randint(0, 255) for _ in range(3)])
        # Center x, y
        buffer = 20
        y = random.randint(buffer, height - buffer - 1)
        x = random.randint(buffer, width - buffer - 1)
        # Size
        s = random.randint(buffer, height // 4)
        return shape, color, (x, y, s)

    def random_image(self, height, width):
        """Creates random specifications of an image with multiple shapes.
        Returns the background color of the image and a list of shape
        specifications that can be used to draw the image.
        """
        def non_max_suppression(boxes, scores, threshold):
            """Performs non-maximum suppression and returns indices of kept boxes.
            boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
            scores: 1-D array of box scores.
            threshold: Float. IoU threshold to use for filtering.
            """
            def compute_iou(box, boxes, box_area, boxes_area):
                """Calculates IoU of the given box with the array of the given boxes.
                box: 1D vector [y1, x1, y2, x2]
                boxes: [boxes_count, (y1, x1, y2, x2)]
                box_area: float. the area of 'box'
                boxes_area: array of length boxes_count.
            
                Note: the areas are passed in rather than calculated here for
                efficiency. Calculate once in the caller to avoid duplicate work.
                """
                # Calculate intersection areas
                y1 = np.maximum(box[0], boxes[:, 0])
                y2 = np.minimum(box[2], boxes[:, 2])
                x1 = np.maximum(box[1], boxes[:, 1])
                x2 = np.minimum(box[3], boxes[:, 3])
                intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
                union = box_area + boxes_area[:] - intersection[:]
                iou = intersection / union
                return iou
            
            assert boxes.shape[0] > 0
            if boxes.dtype.kind != "f":
                boxes = boxes.astype(np.float32)
        
            # Compute box areas
            y1 = boxes[:, 0]
            x1 = boxes[:, 1]
            y2 = boxes[:, 2]
            x2 = boxes[:, 3]
            area = (y2 - y1) * (x2 - x1)
        
            # Get indicies of boxes sorted by scores (highest first)
            ixs = scores.argsort()[::-1]
        
            pick = []
            while len(ixs) > 0:
                # Pick top box and add its index to the list
                i = ixs[0]
                pick.append(i)
                # Compute IoU of the picked box with the rest
                iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
                # Identify boxes with IoU over the threshold. This
                # returns indices into ixs[1:], so add 1 to get
                # indices into ixs.
                remove_ixs = np.where(iou > threshold)[0] + 1
                # Remove indices of the picked and overlapped boxes.
                ixs = np.delete(ixs, remove_ixs)
                ixs = np.delete(ixs, 0)
            return np.array(pick, dtype=np.int32)
        
        # Pick random background color
        bg_color = np.array([random.randint(0, 255) for _ in range(3)])
        # Generate a few random shapes and record their
        # bounding boxes
        shapes = []
        boxes = []
        N = random.randint(1, 4)
        for _ in range(N):
            shape, color, dims = self.random_shape(height, width)
            shapes.append((shape, color, dims))
            x, y, s = dims
            boxes.append([y - s, x - s, y + s, x + s])
        #避免產生的樣本疊加太大，去掉重疊樣本
        keep_ixs = non_max_suppression(
            np.array(boxes), np.arange(N), 0.3)
        shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
        return bg_color, shapes

    def prepare(self, class_map=None):#準備好資料集關聯參數
        def clean_name(name):#顯示類別名
            return ",".join(name.split(",")[:1])

        #產生資料集中，待使用的變數
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self.image_ids = np.arange(self.num_images)




def get_ax(rows=1, cols=1, size=8):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

#訓練資料集dataset
dataset_train = ShapesDataset()
dataset_train.load_shapes(500, mask_rcnn_model.IMAGE_DIM, mask_rcnn_model.IMAGE_DIM)
dataset_train.prepare()

#測試資料集dataset
dataset_val = ShapesDataset()
dataset_val.load_shapes(50, mask_rcnn_model.IMAGE_DIM, mask_rcnn_model.IMAGE_DIM)
dataset_val.prepare()

#載入隨機樣本，並顯示
image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)



BATCH_SIZE =3#8#批次
NUM_CLASSES = 1 + 3  # background + 3 shapes
# Create model in training mode
MODEL_DIR = "./log"
model = MaskRCNN(mode="training", model_dir=MODEL_DIR, num_class=dataset_train.num_classes,batch_size = BATCH_SIZE)#加完背景後81個類別

#模型權重檔案路徑
weights_path = "./mask_rcnn_coco.h5"

#載入權重檔案
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True,exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                            "mrcnn_bbox", "mrcnn_mask"])



model.train(dataset_train, dataset_val,batch_size =  BATCH_SIZE,
            learning_rate=mask_rcnn_model.LEARNING_RATE, 
            epochs=1, 
            layers='heads')

model.train(dataset_train, dataset_val ,batch_size =  BATCH_SIZE,
            learning_rate=mask_rcnn_model.LEARNING_RATE / 10,
            epochs=2, 
            layers="all")
#########################################################################
import os
MODEL_DIR = "mask_model";os.makedirs(MODEL_DIR, exist_ok=True)
model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
model.keras_model.save_weights(model_path)

######################################################################
MODEL_DIR = "mask_model"
model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
#重新建立模型
model2 = MaskRCNN(mode="inference", model_dir=MODEL_DIR, num_class=dataset_train.num_classes,batch_size = 1)#加完背景後81個類別

#載入模型
print("Loading weights from ", model_path)
model2.load_weights(model_path, by_name=True)

#隨機取出圖片
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    mask_rcnn_model.load_image_gt(dataset_val,   image_id, use_mini_mask=False)

ax = get_ax(1, 2)
#顯示原始圖片及標注
visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                            dataset_train.class_names, ax=ax[0])

#使用模型進行預測。並顯示結果
results = model2.detect([original_image], verbose=1)
r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                            dataset_val.class_names, r['scores'], ax=ax[1])

##############################################################
# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(dataset_val.image_ids, 10)
APs = []
for image_id in image_ids:
    #原始圖片
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        mask_rcnn_model.load_image_gt(dataset_val,  image_id, use_mini_mask=False)
    molded_images = np.expand_dims(utils.mold_image(image), 0)
    #執行結果
    results = model2.detect([image], verbose=0)
    r = results[0]
    #計算模型分數
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)
    
print("mAP: ", np.mean(APs))





'''


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



'''

