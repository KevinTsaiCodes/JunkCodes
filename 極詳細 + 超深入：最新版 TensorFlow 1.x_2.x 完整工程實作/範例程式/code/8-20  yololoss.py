# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""
import tensorflow as tf

def _create_mesh_xy(batch_size, grid_h, grid_w, n_box):#產生帶序號的網格
    mesh_x = tf.cast(tf.reshape(tf.tile(tf.range(grid_w), [grid_h]), (1, grid_h, grid_w, 1, 1)),tf.float32)
    mesh_y = tf.transpose(mesh_x, (0,2,1,3,4))
    mesh_xy = tf.tile(tf.concat([mesh_x,mesh_y],-1), [batch_size, 1, 1, n_box, 1])
    return mesh_xy

def adjust_pred_tensor(y_pred):#將網格訊息融入座標，置信度做sigmoid。並重新群組合
    grid_offset = _create_mesh_xy(*y_pred.shape[:4])
    pred_xy    = grid_offset + tf.sigmoid(y_pred[..., :2])  #計算該尺度矩陣上的座標sigma(t_xy) + c_xy
    pred_wh    = y_pred[..., 2:4]                           #取出預測物體的尺寸t_wh
    pred_conf  = tf.sigmoid(y_pred[..., 4])                 #對分類別機率（置信度）做sigmoid轉化
    pred_classes = y_pred[..., 5:]                          #取出分類別結果
    #重新群組合
    preds = tf.concat([pred_xy, pred_wh, tf.expand_dims(pred_conf, axis=-1), pred_classes], axis=-1)
    return preds

#產生一個矩陣。每個格子裡放有3個候選框
def _create_mesh_anchor(anchors, batch_size, grid_h, grid_w, n_box):
    mesh_anchor = tf.tile(anchors, [batch_size*grid_h*grid_w])
    mesh_anchor = tf.reshape(mesh_anchor, [batch_size, grid_h, grid_w, n_box, 2])#每個候選框有2個值
    mesh_anchor = tf.cast(mesh_anchor, tf.float32)
    return mesh_anchor

def conf_delta_tensor(y_true, y_pred, anchors, ignore_thresh):

    pred_box_xy, pred_box_wh, pred_box_conf = y_pred[..., :2], y_pred[..., 2:4], y_pred[..., 4]
    #帶有候選框的格子矩陣
    anchor_grid = _create_mesh_anchor(anchors, *y_pred.shape[:4])#y_pred.shape為（2，13，13，3，15）
    true_wh = y_true[:,:,:,:,2:4]
    true_wh = anchor_grid * tf.exp(true_wh)
    true_wh = true_wh * tf.expand_dims(y_true[:,:,:,:,4], 4)#復原真實尺寸，高和寬
    anchors_ = tf.constant(anchors, dtype='float', shape=[1,1,1,y_pred.shape[3],2])#y_pred.shape[3]為候選框個數
    true_xy = y_true[..., 0:2]#取得中心點
    true_wh_half = true_wh / 2.
    true_mins    = true_xy - true_wh_half#計算起始座標
    true_maxes   = true_xy + true_wh_half#計算尾部座標

    pred_xy = pred_box_xy
    pred_wh = tf.exp(pred_box_wh) * anchors_

    pred_wh_half = pred_wh / 2.
    pred_mins    = pred_xy - pred_wh_half#計算起始座標
    pred_maxes   = pred_xy + pred_wh_half#計算尾部座標

    intersect_mins  = tf.maximum(pred_mins,  true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)

    #計算重疊面積
    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    #計算不重疊面積
    union_areas = pred_areas + true_areas - intersect_areas
    best_ious  = tf.truediv(intersect_areas, union_areas)#計算iou
    #ios小於設定值將作為負向的loss
    conf_delta = pred_box_conf * tf.cast(best_ious < ignore_thresh,tf.float32)
    return conf_delta

def wh_scale_tensor(true_box_wh, anchors, image_size):
    image_size_  = tf.reshape(tf.cast(image_size, tf.float32), [1,1,1,1,2])
    anchors_ = tf.constant(anchors, dtype='float', shape=[1,1,1,3,2])

    #計算高和寬的縮放範圍
    wh_scale = tf.exp(true_box_wh) * anchors_ / image_size_
    #物體尺寸占整個圖片的面積比
    wh_scale = tf.expand_dims(2 - wh_scale[..., 0] * wh_scale[..., 1], axis=4)
    return wh_scale

#位置loss為box之差乘縮放比，所得的結果，再進行平方求和
def loss_coord_tensor(object_mask, pred_box, true_box, wh_scale, xywh_scale):
    xy_delta    = object_mask   * (pred_box-true_box) * wh_scale * xywh_scale
    loss_xy    = tf.reduce_sum(tf.square(xy_delta),       list(range(1,5)))#按照1，2，3，4（xyhw）歸約求和
    return loss_xy

def loss_conf_tensor(object_mask, pred_box_conf, true_box_conf, obj_scale, noobj_scale, conf_delta):
    object_mask_ = tf.squeeze(object_mask, axis=-1)
    conf_delta  = object_mask_ * (pred_box_conf-true_box_conf) * obj_scale + (1-object_mask_) * conf_delta * noobj_scale
    loss_conf  = tf.reduce_sum(tf.square(conf_delta),     list(range(1,4)))#按照1，2，3（候選框）精簡求和，0為批次
    return loss_conf


def loss_class_tensor(object_mask, pred_box_class, true_box_class, class_scale):
    true_box_class_ = tf.cast(true_box_class, tf.int64)
    class_delta = object_mask * \
                  tf.expand_dims(tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_box_class_, logits=pred_box_class), 4) * \
                  class_scale

    loss_class = tf.reduce_sum(class_delta,               list(range(1,5)))
    return loss_class

ignore_thresh=0.5
grid_scale=1
obj_scale=5
noobj_scale=1
xywh_scale=1
class_scale=1
def lossCalculator(y_true, y_pred, anchors,image_size): #image_size【h,w】
    y_pred = tf.reshape(y_pred, y_true.shape) #(2, 13, 13, 3, 15)

    object_mask = tf.expand_dims(y_true[..., 4], 4)#(2, 13, 13, 3, 1)
    preds = adjust_pred_tensor(y_pred)#將box與置信度數值變化後重新群組合

    conf_delta = conf_delta_tensor(y_true, preds, anchors, ignore_thresh)
    wh_scale =  wh_scale_tensor(y_true[..., 2:4], anchors, image_size)

    loss_box = loss_coord_tensor(object_mask, preds[..., :4], y_true[..., :4], wh_scale, xywh_scale)
    loss_conf = loss_conf_tensor(object_mask, preds[..., 4], y_true[..., 4], obj_scale, noobj_scale, conf_delta)
    loss_class = loss_class_tensor(object_mask, preds[..., 5:], y_true[..., 5:], class_scale)
    loss = loss_box + loss_conf + loss_class
    return loss*grid_scale

def loss_fn(list_y_trues, list_y_preds,anchors,image_size):
    inputanchors = [anchors[12:],anchors[6:12],anchors[:6]]
    losses = [lossCalculator(list_y_trues[i], list_y_preds[i], inputanchors[i],image_size) for i in range(len(list_y_trues)) ]
    return tf.sqrt(tf.reduce_sum(losses)) #將三個矩陣的loss相加再開平方


