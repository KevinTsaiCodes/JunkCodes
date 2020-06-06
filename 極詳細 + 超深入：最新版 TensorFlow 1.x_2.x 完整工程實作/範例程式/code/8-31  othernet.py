# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室 
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""
import tensorflow as tf
from tensorflow.keras import layers as KL
from tensorflow.keras import models as KM
from tensorflow.keras import backend as K #載入keras的後端實現
import numpy as np
utils = __import__("8-30  mask_rcnn_utils")
mask_rcnn_model = __import__("8-29  mask_rcnn_model")

############################################################
#  Region Proposal Network (RPN)
############################################################
#建構RPN網路圖結構，一共分為兩部分：1計算分數，2計算邊框
def rpn_graph(feature_map,#輸入的特征，其w與h所圍成面積的個數當作錨點的個數。
              anchors_per_location, #每個待計算錨點的網格，需要劃分幾種形狀的矩形
              anchor_stride):#掃描網格的步長
    
    #透過一個卷冊積得到共享特征
    shared = KL.Conv2D(512, (3, 3), padding='same', activation='relu',
                       strides=anchor_stride,name='rpn_conv_shared')(feature_map)

    #第一部分計算錨點的分數（前景和背景） [batch, height, width, anchors per location * 2].
    x = KL.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                  activation='linear', name='rpn_class_raw')(shared)

    #將feature_map展開，得到[batch, anchors, 2]。anchors=feature_map的h*w*anchors_per_location 
    rpn_class_logits = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

    #用Softmax來分類別前景和背景BG/FG.結果當作分數
    rpn_probs = KL.Activation(
        "softmax", name="rpn_class_xxx")(rpn_class_logits)

    #第二部分計算錨點的邊框，每個網格劃分anchors_per_location種矩形框，每種4個座標
    x = KL.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
                  activation='linear', name='rpn_bbox_pred')(shared)

    #將feature_map展開，得到[batch, anchors, 4]
    rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]


def build_rpn_model(anchor_stride, #掃描網格的步長
                    anchors_per_location, #每個待計算錨點的網格，需要劃分幾種形狀的矩形
                    depth):               #輸入的特征有多少個

    input_feature_map = KL.Input(shape=[None, None, depth],name="input_rpn_feature_map")
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
    return KM.Model([input_feature_map], outputs, name="rpn_model")


############################################################
#  Proposal Layer
############################################################
#按照指定的框與偏移量，計算最終的框
def apply_box_deltas_graph(boxes, #[N, (y1, x1, y2, x2)]
                           deltas):#[N, (dy, dx, log(dh), log(dw))] 
    
    #轉換成中心點和h，w格式
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    #計算偏移
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    #轉成左上，右下兩個點 y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name="apply_box_deltas_out")
    return result

#將框座標限制在0到1之間
def clip_boxes_graph(boxes, #計算完的box[N, (y1, x1, y2, x2)]
                     window):#y1, x1, y2, x2[0, 0, 1, 1]

    # Split
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped


class ProposalLayer(tf.keras.layers.Layer):#RPN最終處理層

    def __init__(self, proposal_count, nms_threshold,batch_size, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold
        self.batch_size = batch_size


    def call(self, inputs):
        '''
        輸入字段input描述
        rpn_probs: [batch, num_anchors, 2] #(bg機率, fg機率)
        rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
        anchors: [batch, (y1, x1, y2, x2)] 
        '''
        #將前景機率值取出[Batch, num_anchors, 1]
        scores = inputs[0][:, :, 1]
        #取出位置偏移量[batch, num_anchors, 4]
        deltas = inputs[1]
        deltas = deltas * np.reshape(mask_rcnn_model.RPN_BBOX_STD_DEV, [1, 1, 4])
        #取出錨點 Anchors
        anchors = inputs[2]

        #獲得前6000個分值最大的資料
        pre_nms_limit = tf.minimum(6000, tf.shape(anchors)[1])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True, name="top_anchors").indices
        #取得scores中索引為ix的值
        scores = utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y),self.batch_size)
        deltas = utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y),self.batch_size)
        pre_nms_anchors = utils.batch_slice([anchors, ix], lambda a, x: tf.gather(a, x),
                                    self.batch_size, names=["pre_nms_anchors"])

        #得出最終的框座標。[batch, N,4] (y1, x1, y2, x2),將框按照偏移縮放的資料進行計算，
        boxes = utils.batch_slice([pre_nms_anchors, deltas],
                                  lambda x, y: apply_box_deltas_graph(x, y),self.batch_size,
                                  names=["refined_anchors"])


        #對出界的box進行剪輯，範圍控制在 0.到1 [batch, N, (y1, x1, y2, x2)]
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = utils.batch_slice(boxes,lambda x: clip_boxes_graph(x, window), self.batch_size,
                                  names=["refined_anchors_clipped"])

        # Non-max suppression算法
        def nms(boxes, scores):
            indices = tf.image.non_max_suppression(boxes, scores, self.proposal_count,
                self.nms_threshold, name="rpn_non_max_suppression")#計算nms，並獲得索引
            proposals = tf.gather(boxes, indices)#在boxes中取出indices索引所指的值
            #若果proposals的個數小於proposal_count，剩下的補0
            padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals
        proposals = utils.batch_slice([boxes, scores], nms,self.batch_size)
        return proposals

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 4)



############################################################
#  ROIAlign Layer
############################################################


#PyramidROIAlign處理
class PyramidROIAlign(tf.keras.layers.Layer):

    def __init__(self,batch_size, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)
        self.batch_size = batch_size
        
    def log2_graph(self, x):#計算log2
        return tf.log(x) / tf.log(2.0)
    
    def call(self, inputs):
        '''
        輸導入參數數 Inputs:
        -ROIboxes(RPN結果): [batch, num_boxes, 4]，4：(y1, x1, y2, x2)。nms後得錨點座標.num_boxes=1000
        - image_meta: [batch, (meta data)] 圖片的附加訊息 93
        - Feature maps: [P2, P3, P4, P5]骨干網經由fpn後的特征.每個[batch, height, width, channels]
        [(1, 256, 256, 256),(1, 128, 128, 256),(1, 64, 64, 256),(1, 32, 32, 256)]
        '''
        #取得輸導入參數數
        ROIboxes = inputs[0]#(1, 1000, 4) 
        image_meta = inputs[1]#(1, 93)
        feature_maps = inputs[2:]

        #將錨點座標提出來
        y1, x1, y2, x2 = tf.split(ROIboxes, 4, axis=2)#[batch, num_boxes, 4]
        h = y2 - y1
        w = x2 - x1

        
        ###############################在這1000個ROI裡，按固定算法比對到不同level的特征。
        #獲得圖片形狀
        image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        #因為h與w是標准化座標。其分母已經被除了tf.sqrt(image_area)。
        #這裡再除以tf.sqrt(image_area)分之1，是為了變為像素座標
        roi_level = self.log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum( 2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        # 每個roi按照自己的區域去對應的特征裡截取內容，並resize成特殊的7*7大小. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            #equal會傳回一個true false的（1，1000），where傳回其中為true的索引[[0,1],[0,4],,,[0,200]]
            ix = tf.where(tf.equal(roi_level, level),name="ix")#(828, 2)

            
            #在多維上建立索引取值[?,4](828, 4)
            level_boxes = tf.gather_nd(ROIboxes, ix,name="level_boxes")#在(1, 1000, 4)上按照[[0,1],[0,4],,,[0,200]]取值 

            # Box indices for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)#(828, )，【0，0，0，0，0，】若果批次為2，就是[000...111]

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            #下面兩個值，是ROIboxes中按照不同尺度劃分好的索引，對於該尺度特征中的批次索引，不希望有變化。所以停止梯度
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)


            #結果: [batch * num_boxes, pool_height, pool_width, channels]
            #feature_maps [(1, 256, 256, 256),(1, 128, 128, 256),(1, 64, 64, 256),(1, 32, 32, 256)]
            #box_indices一共level_boxes個。指定level_boxes中的第幾個框，作用於feature_maps中的第幾個圖片
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, self.pool_shape, method="bilinear"))

        #1000個roi都取到了對應的內容，將它們群組合起來。( 1000, 7, 7, 256) 
        pooled = tf.concat(pooled, axis=0)#其中的順序是按照level來的需要重新排序成原來ROIboxes順序

        #重新排序成原來ROIboxes順序
        box_to_level = tf.concat(box_to_level, axis=0)#按照選取level的順序
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],axis=1)#[1000，3] 3([xi] range)
                                 

        #取出頭兩個批次+序號（1000個），每個值代表原始roi展開的索引了。
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1] #確保一個批次在100000以內
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(#按照索引排序，
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)#將roi中的順序對應到pooled中的索引取出來
        pooled = tf.gather(pooled, ix)#按照索引從pooled中取出的框，就是原始順序了。

        #加上批次維度，並傳回
        pooled = tf.expand_dims(pooled, 0)#應該用reshape
        #pooled = KL.Reshape([self.batch_size,-1, self.pool_shape, self.pool_shape, mask_rcnn_model.FPN_FEATURE], name="pooled")(pooled)
        #pooled = tf.reshape(pooled, [self.batch_size,1000, self.pool_shape, self.pool_shape, mask_rcnn_model.FPN_FEATURE])
        return pooled
    
    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1], )
    
############################################################
#  Feature Pyramid Network Heads
############################################################

def fpn_classifier_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, batch_size, train_bn=True,
                         fc_layers_size=1024):


    #ROIAlign層 Shape: [batch, num_boxes, pool_height, pool_width, channels]
    x = PyramidROIAlign(batch_size,[pool_size, pool_size],
                        name="roi_align_classifier")([rois, image_meta] + feature_maps)
    #用卷冊積替代兩個1024全連線網路
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"),
                           name="mrcnn_class_conv1")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_class_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    #1*1卷冊積，代替第二個全連線
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (1, 1)), name="mrcnn_class_conv2")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_class_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    
    #共享特征，用於計算分類別和邊框
    shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2),name="pool_squeeze")(x)

    #（1）計算分類別
    mrcnn_class_logits = KL.TimeDistributed(KL.Dense(num_classes),
                                            name='mrcnn_class_logits')(shared)
    mrcnn_probs = KL.TimeDistributed(KL.Activation("softmax"),
                                     name="mrcnn_class")(mrcnn_class_logits)

    #（2）計算邊框座標BBox（偏移和縮放量）
    # [batch, boxes, num_classes * (dy, dx, log(dh), log(dw))]
    x = KL.TimeDistributed(KL.Dense(num_classes * 4, activation='linear'),
                           name='mrcnn_bbox_fc')(shared)
    # Reshape to [batch, boxes, num_classes, (dy, dx, log(dh), log(dw))]
    s = K.int_shape(x)
    print(s, num_classes, 4)
    #mrcnn_bbox = KL.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)
    mrcnn_bbox = KL.Reshape((-1, num_classes, 4), name="mrcnn_bbox")(x)


    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox    
 
############################################################
#  Detection Layer
############################################################
#實物邊框檢驗，傳回最終的標准化區域座標[batch, num_detections, (y1, x1, y2, x2, class_id, class_score)]
class DetectionLayer(tf.keras.layers.Layer):

    def __init__(self,batch_size,  **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.batch_size = batch_size
        
    def call(self, inputs):#輸入：rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta
        #分析參數
        rois,mrcnn_class,mrcnn_bbox,image_meta = inputs

        #解析圖片附加訊息
        m = parse_image_meta_graph(image_meta)
        image_shape = m['image_shape'][0]
        #window為pading後，真實圖片的像素座標，將其轉化為標准座標
        window = norm_boxes_graph(m['window'], image_shape[:2])
        
        #根據分類別訊息，對原始roi進行再一次的過濾。得到DETECTION_MAX_INSTANCES個。不足的補0
        detections_batch = utils.batch_slice(
            [rois, mrcnn_class, mrcnn_bbox, window],
            lambda x, y, w, z: refine_detections_graph(x, y, w, z),
            self.batch_size)

        #將標准化座標及過濾後的結果 reshape後傳回。
        return tf.reshape(
            detections_batch,
            [self.batch_size, mask_rcnn_model.DETECTION_MAX_INSTANCES, 6])

    def compute_output_shape(self, input_shape):
        return (None, mask_rcnn_model.DETECTION_MAX_INSTANCES, 6)

#將座標按照圖片大小，轉化為標准化座標
def norm_boxes_graph(boxes, #像素座標(y1, x1, y2, x2)
                     shape):#像素邊長(height, width)
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)#標准化座標[..., (y1, x1, y2, x2)]

#分類別器結果的最終處理函數，傳回剪輯後的標准座標與去重後的分類別結果
def refine_detections_graph(rois, probs, deltas, window):
   
    #取出每個ROI的 Class IDs
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
    
    #取出每個ROI的class 索引
    indices = tf.stack([tf.range(tf.shape(probs)[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(probs, indices)#根據索引獲得分數
    
    deltas_specific = tf.gather_nd(deltas, indices)#根據索引獲得box區域座標(待修正的偏差)

    #將偏差套用到rois框中
    refined_rois = apply_box_deltas_graph( rois, deltas_specific * mask_rcnn_model.BBOX_STD_DEV)
    #對出界的框進行剪輯
    refined_rois = clip_boxes_graph(refined_rois, window)

    #取出前景的類別索引（將背景類別過濾掉）
    keep = tf.where(class_ids > 0)[:, 0]
    #從前景類別裡，再將分數小於DETECTION_MIN_CONFIDENCE的過濾掉
    if mask_rcnn_model.DETECTION_MIN_CONFIDENCE:
        conf_keep = tf.where(class_scores >= mask_rcnn_model.DETECTION_MIN_CONFIDENCE)[:, 0]
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(conf_keep, 0))
        keep = tf.sparse_tensor_to_dense(keep)[0]

    #根據剩下的keep索引取出對應的值
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois,   keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):#定義nms函數，對每個類別做去重
        
        #找出類別別為class_id 的索引
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        
        #對該類別的roi按照設定值DETECTION_NMS_THRESHOLD進行區域去重，最多獲得DETECTION_MAX_INSTANCES個結果，
        class_keep = tf.image.non_max_suppression(
                tf.gather(pre_nms_rois, ixs),
                tf.gather(pre_nms_scores, ixs),
                max_output_size=mask_rcnn_model.DETECTION_MAX_INSTANCES,
                iou_threshold=mask_rcnn_model.DETECTION_NMS_THRESHOLD)
        #將去重後的索引轉化為roi中的索引
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        #資料對齊，當去重後的個數小於DETECTION_MAX_INSTANCES時，對其補-1.
        gap = mask_rcnn_model.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)],
                            mode='CONSTANT', constant_values=-1)
        #將形狀統一變為[mask_rcnn_model.DETECTION_MAX_INSTANCES]，並傳回
        class_keep.set_shape([mask_rcnn_model.DETECTION_MAX_INSTANCES])
        return class_keep

    #對每個class IDs做去重動作。
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
                         dtype=tf.int64)
    #將list結果中的元素合並到一個陣列裡。並刪掉-1的值
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
    #計算交集。沒用
#    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
#                                    tf.expand_dims(nms_keep, 0))
#    keep = tf.sparse_tensor_to_dense(keep)[0]
    keep = nms_keep#改成這樣
    #nms之後，根據剩下的keep索引取出對應的值，將總是控制在DETECTION_MAX_INSTANCES之內
    roi_count = mask_rcnn_model.DETECTION_MAX_INSTANCES
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)#keep個數小於DETECTION_MAX_INSTANCES


    #拼接輸出結果[N, (y1, x1, y2, x2, class_id, score)]
    detections = tf.concat([ tf.gather(refined_rois, keep),
        tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
        ], axis=1)

    #資料對齊，不足DETECTION_MAX_INSTANCES的補0，並傳回。
    gap = mask_rcnn_model.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    return detections    

#語義分割
def build_fpn_mask_graph(rois,#目的實物檢驗結果，標准座標[batch, num_rois, (y1, x1, y2, x2)] 
                         feature_maps,#骨干網之後的fpn特征[P2, P3, P4, P5]
                         image_meta,
                         pool_size, num_classes,batch_size, train_bn=True):
    """
    傳回: Masks [batch, roi_count, height, width, num_classes]
    """
    #ROIAlign 最終統一池化的大小為14 
    # Shape: [batch, boxes, pool_height, pool_width, channels]
    x = PyramidROIAlign(batch_size,[pool_size, pool_size],
                        name="roi_align_mask")([rois, image_meta] + feature_maps)

    # Conv layers
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv1")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_mask_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv2")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_mask_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv3")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_mask_bn3')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv4")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_mask_bn4')(x, training=train_bn)
    x = KL.Activation('relu')(x)#(1, ?, 14, 14, 256)
    
    #使用反卷冊積進行上取樣
    x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
                           name="mrcnn_mask_deconv")(x)#(1, ?, 28, 28, 256)
    #用卷冊積代替全連線
    x = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"),
                           name="mrcnn_mask")(x)
    return x
























############################################################
#  Data Formatting
############################################################



def parse_image_meta(meta):
    """Parses an array that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES

    Returns a dict of the parsed values.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id.astype(np.int32),
        "original_image_shape": original_image_shape.astype(np.int32),
        "image_shape": image_shape.astype(np.int32),
        "window": window.astype(np.int32),
        "scale": scale.astype(np.float32),
        "active_class_ids": active_class_ids.astype(np.int32),
    }


def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES

    Returns a dict of the parsed tensors.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id,
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "window": window,
        "scale": scale,
        "active_class_ids": active_class_ids,
    }

def trim_zeros_graph(boxes, name=None):
    """Often boxes are represented with matrices of shape [N, 4] and
    are padded with zeros. This removes zero boxes.

    boxes: [N, 4] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros
