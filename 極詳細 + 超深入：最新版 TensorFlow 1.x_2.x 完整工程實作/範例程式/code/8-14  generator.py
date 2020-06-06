# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室 
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""
import numpy as np
from random import shuffle
annotation = __import__("8-13  annotation")
parse_annotation = annotation.parse_annotation
ImgAugment= annotation.ImgAugment

box = __import__("8-15  box")
find_match_box = box.find_match_box

DOWNSAMPLE_RATIO = 32

class BatchGenerator(object):
    def __init__(self, ann_fnames, img_dir,labels,
                 batch_size, anchors,   net_size=416,
                 jitter=True, shuffle=True):
        self.ann_fnames = ann_fnames
        self.img_dir = img_dir
        self.lable_names = labels
        self._net_size = net_size
        self.jitter = jitter
        self.anchors = create_anchor_boxes(anchors)#按照anchors的尺寸，產生座標從00開始的框
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps_per_epoch = int(len(ann_fnames) / batch_size)
        self._epoch = 0
        self._end_epoch = False
        self._index = 0

    def next_batch(self):
        xs,ys_1,ys_2,ys_3 = [],[],[],[]
        for _ in range(self.batch_size):
            x, y1, y2, y3 = self._get()
            xs.append(x)
            ys_1.append(y1)
            ys_2.append(y2)
            ys_3.append(y3)
        if self._end_epoch == True:
            if self.shuffle:
                shuffle(self.ann_fnames)
            self._end_epoch = False
            self._epoch += 1
        return np.array(xs).astype(np.float32), np.array(ys_1).astype(np.float32), np.array(ys_2).astype(np.float32), np.array(ys_3).astype(np.float32)

    def _get(self):
        net_size = self._net_size
        #解析標注檔案
        fname, boxes, coded_labels = parse_annotation(self.ann_fnames[self._index], self.img_dir, self.lable_names)
        #讀取圖片，並按照設定修改圖片尺寸
        img_augmenter = ImgAugment(net_size, net_size, self.jitter)
        img, boxes_ = img_augmenter.imread(fname, boxes)

        #產生3種尺度的格子
        list_ys = _create_empty_xy(net_size, len(self.lable_names))
        for original_box, label in zip(boxes_, coded_labels):
            #在anchors中，找到與其面積區域最比對的候選框max_anchor，對應的尺度索引，該尺度下的第幾個錨點
            max_anchor, scale_index, box_index = _find_match_anchor(original_box, self.anchors)
            #計算在對應尺度上的中心點座標和對應候選框的長寬縮放比例
            _coded_box = _encode_box(list_ys[scale_index], original_box, max_anchor, net_size, net_size)
            _assign_box(list_ys[scale_index], box_index, _coded_box, label)

        self._index += 1
        if self._index == len(self.ann_fnames):
            self._index = 0
            self._end_epoch = True
        
        return img/255., list_ys[2], list_ys[1], list_ys[0]

#起始化標簽
def _create_empty_xy(net_size, n_classes, n_boxes=3):
    #獲得最小矩陣格子
    base_grid_h, base_grid_w = net_size//DOWNSAMPLE_RATIO, net_size//DOWNSAMPLE_RATIO
    #起始化三種不同尺度的矩陣。用於存放標簽
    ys_1 = np.zeros((1*base_grid_h,  1*base_grid_w, n_boxes, 4+1+n_classes)) 
    ys_2 = np.zeros((2*base_grid_h,  2*base_grid_w, n_boxes, 4+1+n_classes)) 
    ys_3 = np.zeros((4*base_grid_h,  4*base_grid_w, n_boxes, 4+1+n_classes)) 
    list_ys = [ys_3, ys_2, ys_1]
    return list_ys

def _encode_box(yolo, original_box, anchor_box, net_w, net_h):
    x1, y1, x2, y2 = original_box
    _, _, anchor_w, anchor_h = anchor_box
    #取出格子在高和寬方向上的個數
    grid_h, grid_w = yolo.shape[:2]
    
    #根據原始圖片到目前矩陣的縮放比例，計算目前矩陣中，物體的中心點座標
    center_x = .5*(x1 + x2)
    center_x = center_x / float(net_w) * grid_w 
    center_y = .5*(y1 + y2)
    center_y = center_y / float(net_h) * grid_h
    
    #計算物體相對於候選框的尺寸縮放值
    w = np.log(max((x2 - x1), 1) / float(anchor_w)) # t_w
    h = np.log(max((y2 - y1), 1) / float(anchor_h)) # t_h
    box = [center_x, center_y, w, h]#將中心點和縮放值打包傳回
    return box

#找到與物體尺寸最接近的候選框
def _find_match_anchor(box, anchor_boxes):
    x1, y1, x2, y2 = box
    shifted_box = np.array([0, 0, x2-x1, y2-y1])
    max_index = find_match_box(shifted_box, anchor_boxes)
    max_anchor = anchor_boxes[max_index]
    scale_index = max_index // 3
    box_index = max_index%3
    return max_anchor, scale_index, box_index

#將實際的值放到標簽矩陣裡。作為真正的標簽
def _assign_box(yolo, box_index, box, label):
    center_x, center_y, _, _ = box
    #向下取整，得到的就是格子的索引
    grid_x = int(np.floor(center_x))
    grid_y = int(np.floor(center_y))
    #填入所計算的數值，作為標簽
    yolo[grid_y, grid_x, box_index]      = 0.
    yolo[grid_y, grid_x, box_index, 0:4] = box
    yolo[grid_y, grid_x, box_index, 4  ] = 1.
    yolo[grid_y, grid_x, box_index, 5+label] = 1.

def create_anchor_boxes(anchors):#將候選框變為box
    boxes = []
    n_boxes = int(len(anchors)/2)
    for i in range(n_boxes):
        boxes.append(np.array([0, 0, anchors[2*i], anchors[2*i+1]]))
    return np.array(boxes)




