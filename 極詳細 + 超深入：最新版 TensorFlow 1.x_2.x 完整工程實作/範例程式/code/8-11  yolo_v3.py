# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室 
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""
import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

#定義darknet塊：一個短連結加一個同尺度卷冊積再加一個下取樣卷冊積
def _darknet53_block(inputs, filters):
    shortcut = inputs
    inputs = slim.conv2d(inputs, filters, 1, stride=1, padding='SAME')#標準卷冊積
    inputs = slim.conv2d(inputs, filters * 2, 3, stride=1, padding='SAME')#標準卷冊積

    inputs = inputs + shortcut
    return inputs


def _conv2d_fixed_padding(inputs, filters, kernel_size, strides=1):
    assert strides>1

    inputs = _fixed_padding(inputs, kernel_size)#外圍填充0，好支援valid卷冊積
    inputs = slim.conv2d(inputs, filters, kernel_size, stride=strides, padding= 'VALID')

    return inputs

#對指定輸入填充0
def _fixed_padding(inputs, kernel_size, *args, mode='CONSTANT', **kwargs):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    #inputs 【b,h,w,c】  pad  b,c不變。h和w上下左右，填充0.kernel = 3 ，則上下左右各加一趟0
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]], mode=mode)
    return padded_inputs

#定義Darknet-53 模型.傳回3個不同尺度的特征
def darknet53(inputs):
    inputs = slim.conv2d(inputs, 32, 3, stride=1, padding='SAME')#標準卷冊積
    inputs = _conv2d_fixed_padding(inputs, 64, 3, strides=2)#需要填充,並使用了'VALID' (-1, 208, 208, 64)
    
    inputs = _darknet53_block(inputs, 32)#darknet塊
    inputs = _conv2d_fixed_padding(inputs, 128, 3, strides=2)

    for i in range(2):
        inputs = _darknet53_block(inputs, 64)
    inputs = _conv2d_fixed_padding(inputs, 256, 3, strides=2)

    for i in range(8):
        inputs = _darknet53_block(inputs, 128)
    route_1 = inputs  #特征1 (-1, 52, 52, 128)

    inputs = _conv2d_fixed_padding(inputs, 512, 3, strides=2)
    for i in range(8):
        inputs = _darknet53_block(inputs, 256)
    route_2 = inputs#特征2  (-1, 26, 26, 256)

    inputs = _conv2d_fixed_padding(inputs, 1024, 3, strides=2)
    for i in range(4):
        inputs = _darknet53_block(inputs, 512)#特征3 (-1, 13, 13, 512)

    return route_1, route_2, inputs#在原有的darknet53，還會跟一個全局池化。這裡沒有使用。所以其實是只有52層




_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1

#定義候選框，來自coco資料集
_ANCHORS = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]

#yolo檢驗塊
def _yolo_block(inputs, filters):
    inputs = slim.conv2d(inputs, filters, 1, stride=1, padding='SAME')#標準卷冊積
    inputs = slim.conv2d(inputs, filters * 2, 3, stride=1, padding='SAME')#標準卷冊積
    inputs = slim.conv2d(inputs, filters, 1, stride=1, padding='SAME')#標準卷冊積
    inputs = slim.conv2d(inputs, filters * 2, 3, stride=1, padding='SAME')#標準卷冊積 
    inputs = slim.conv2d(inputs, filters, 1, stride=1, padding='SAME')#標準卷冊積
    route = inputs
    inputs = slim.conv2d(inputs, filters * 2, 3, stride=1, padding='SAME')#標準卷冊積 
    return route, inputs

#檢驗層
def _detection_layer(inputs, num_classes, anchors, img_size, data_format):
    print(inputs.get_shape())
    num_anchors = len(anchors)#候選框個數
    predictions = slim.conv2d(inputs, num_anchors * (5 + num_classes), 1, stride=1, normalizer_fn=None,
                              activation_fn=None, biases_initializer=tf.zeros_initializer())

    shape = predictions.get_shape().as_list()
    print("shape",shape)#三個尺度的形狀分別為：[1, 13, 13, 3*(5+c)]、[1, 26, 26, 3*(5+c)]、[1, 52, 52, 3*(5+c)]
    grid_size = shape[1:3]#去 NHWC中的HW
    dim = grid_size[0] * grid_size[1]#每個格子所包括的像素
    bbox_attrs = 5 + num_classes

    predictions = tf.reshape(predictions, [-1, num_anchors * dim, bbox_attrs])#把h和w展開成dim

    stride = (img_size[0] // grid_size[0], img_size[1] // grid_size[1])#縮放參數 32（416/13）

    anchors = [(a[0] / stride[0], a[1] / stride[1]) for a in anchors]#將候選框的尺寸同比例拉遠

    #將包括邊框的單元屬性分割
    box_centers, box_sizes, confidence, classes = tf.split(predictions, [2, 2, 1, num_classes], axis=-1)

    box_centers = tf.nn.sigmoid(box_centers)
    confidence = tf.nn.sigmoid(confidence)

    grid_x = tf.range(grid_size[0], dtype=tf.float32)#定義網格索引0,1,2...n
    grid_y = tf.range(grid_size[1], dtype=tf.float32)#定義網格索引0,1,2,...m
    a, b = tf.meshgrid(grid_x, grid_y)#產生網格矩陣 a0，a1.。。an（共M行）  ， b0，b0，。。。b0（共n個），第二行為b1

    x_offset = tf.reshape(a, (-1, 1))#展開 一共dim個
    y_offset = tf.reshape(b, (-1, 1))

    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)#連線----[dim,2]
    x_y_offset = tf.reshape(tf.tile(x_y_offset, [1, num_anchors]), [1, -1, 2])#按候選框的個數複製xy（【1，n】代表第0維一次，第1維n次）

    box_centers = box_centers + x_y_offset#box_centers為0-1，x_y為實際網格的索引，相加後，就是真實位置(0.1+4=4.1，第4個網格裡0.1的偏移)
    box_centers = box_centers * stride#真實尺寸像素點

    anchors = tf.tile(anchors, [dim, 1])
    box_sizes = tf.exp(box_sizes) * anchors#計算邊長：hw
    box_sizes = box_sizes * stride#真實邊長

    detections = tf.concat([box_centers, box_sizes, confidence], axis=-1)
    classes = tf.nn.sigmoid(classes)
    predictions = tf.concat([detections, classes], axis=-1)#將轉化後的結果合起來
    print(predictions.get_shape())#三個尺度的形狀分別為：[1, 507（13*13*3）, 5+c]、[1, 2028, 5+c]、[1, 8112, 5+c]
    return predictions#傳回預測值

#定義上取樣函數
def _upsample(inputs, out_shape):
    #由於上取樣的填充模式不同，tf.image.resize_bilinear會對結果影響很大
    inputs = tf.image.resize_nearest_neighbor(inputs, (out_shape[1], out_shape[2]))
    inputs = tf.identity(inputs, name='upsampled')
    return inputs


#定義yolo函數
def yolo_v3(inputs, num_classes, is_training=False, data_format='NHWC', reuse=False):

    assert data_format=='NHWC'
    
    img_size = inputs.get_shape().as_list()[1:3]#獲得輸入圖片大小

    inputs = inputs / 255    #歸一化

    #定義批次歸一化參數
    batch_norm_params = {
        'decay': _BATCH_NORM_DECAY,
        'epsilon': _BATCH_NORM_EPSILON,
        'scale': True,
        'is_training': is_training,
        'fused': None,  
    }

    #定義yolo網路.
    with slim.arg_scope([slim.conv2d, slim.batch_norm], data_format=data_format, reuse=reuse):
        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params,
                            biases_initializer=None, activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=_LEAKY_RELU)):
            with tf.variable_scope('darknet-53'):
                route_1, route_2, inputs = darknet53(inputs)

            with tf.variable_scope('yolo-v3'):
                route, inputs = _yolo_block(inputs, 512)#(-1, 13, 13, 1024)
                #使用候選框參數來輔助識別
                detect_1 = _detection_layer(inputs, num_classes, _ANCHORS[6:9], img_size, data_format)
                detect_1 = tf.identity(detect_1, name='detect_1')

                
                inputs = slim.conv2d(route, 256, 1, stride=1, padding='SAME')#標準卷冊積 
                upsample_size = route_2.get_shape().as_list()
                inputs = _upsample(inputs, upsample_size)
                inputs = tf.concat([inputs, route_2], axis=3)

                route, inputs = _yolo_block(inputs, 256)#(-1, 26, 26, 512)
                detect_2 = _detection_layer(inputs, num_classes, _ANCHORS[3:6], img_size, data_format)
                detect_2 = tf.identity(detect_2, name='detect_2')

                inputs = slim.conv2d(route, 128, 1, stride=1, padding='SAME')#標準卷冊積
                upsample_size = route_1.get_shape().as_list()
                inputs = _upsample(inputs, upsample_size)
                inputs = tf.concat([inputs, route_1], axis=3)

                _, inputs = _yolo_block(inputs, 128)#(-1, 52, 52, 256)

                detect_3 = _detection_layer(inputs, num_classes, _ANCHORS[0:3], img_size, data_format)
                detect_3 = tf.identity(detect_3, name='detect_3')

                detections = tf.concat([detect_1, detect_2, detect_3], axis=1)
                detections = tf.identity(detections, name='detections')
                return detections#傳回了3個尺度。每個尺度裡又包括3個結果(-1, 10647（ 507 +2028 + 8112）, 5+c)




'''--------Test the scale--------'''
if __name__ == "__main__":
    tf.reset_default_graph()
    import cv2
    data = cv2.imread(  'timg.jpg' )
    data = cv2.cvtColor( data, cv2.COLOR_BGR2RGB )
    data = cv2.resize( data, ( 416, 416 ) )

    data = tf.cast( tf.expand_dims( tf.constant( data ), 0 ), tf.float32 )

    detections = yolo_v3( data,3,data_format='NHWC' )

    with tf.Session() as sess:

        sess.run( tf.global_variables_initializer() )

        print( sess.run( detections ).shape )