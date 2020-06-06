# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
yolo_model = __import__("8-11  yolo_v3")
yolo_v3 = yolo_model.yolo_v3

size = 416
input_img ='timg.jpg'
output_img = 'out.jpg'
class_names = 'coco.names'
weights_file = 'yolov3.weights'
conf_threshold = 0.5 #置信度設定值
iou_threshold = 0.4  #重疊區域設定值

#定義函數：將中心點、高、寬座標 轉化為[x0, y0, x1, y1]座標形式
def detections_boxes(detections):
    center_x, center_y, width, height, attrs = tf.split(detections, [1, 1, 1, 1, -1], axis=-1)
    w2 = width / 2
    h2 = height / 2
    x0 = center_x - w2
    y0 = center_y - h2
    x1 = center_x + w2
    y1 = center_y + h2

    boxes = tf.concat([x0, y0, x1, y1], axis=-1)
    detections = tf.concat([boxes, attrs], axis=-1)
    return detections

#定義函數計算兩個框的內定重疊情況（IOU）box1，box2為左上、右下的座標[x0, y0, x1, x2]
def _iou(box1, box2):

    b1_x0, b1_y0, b1_x1, b1_y1 = box1
    b2_x0, b2_y0, b2_x1, b2_y1 = box2

    int_x0 = max(b1_x0, b2_x0)
    int_y0 = max(b1_y0, b2_y0)
    int_x1 = min(b1_x1, b2_x1)
    int_y1 = min(b1_y1, b2_y1)

    int_area = (int_x1 - int_x0) * (int_y1 - int_y0)

    b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
    b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)

    #分母加個1e-05，避免除數為 0
    iou = int_area / (b1_area + b2_area - int_area + 1e-05)
    return iou






#使用NMS方法，對結果去重
def non_max_suppression(predictions_with_boxes, confidence_threshold, iou_threshold=0.4):

    conf_mask = np.expand_dims((predictions_with_boxes[:, :, 4] > confidence_threshold), -1)
    predictions = predictions_with_boxes * conf_mask

    result = {}
    for i, image_pred in enumerate(predictions):
        shape = image_pred.shape
        print("shape1",shape)
        non_zero_idxs = np.nonzero(image_pred)
        image_pred = image_pred[non_zero_idxs[0]]
        print("shape2",image_pred.shape)
        image_pred = image_pred.reshape(-1, shape[-1])

        bbox_attrs = image_pred[:, :5]
        classes = image_pred[:, 5:]
        classes = np.argmax(classes, axis=-1)

        unique_classes = list(set(classes.reshape(-1)))

        for cls in unique_classes:
            cls_mask = classes == cls
            cls_boxes = bbox_attrs[np.nonzero(cls_mask)]
            cls_boxes = cls_boxes[cls_boxes[:, -1].argsort()[::-1]]
            cls_scores = cls_boxes[:, -1]
            cls_boxes = cls_boxes[:, :-1]

            while len(cls_boxes) > 0:
                box = cls_boxes[0]
                score = cls_scores[0]
                if not cls in result:
                    result[cls] = []
                result[cls].append((box, score))
                cls_boxes = cls_boxes[1:]
                ious = np.array([_iou(box, x) for x in cls_boxes])
                iou_mask = ious < iou_threshold
                cls_boxes = cls_boxes[np.nonzero(iou_mask)]
                cls_scores = cls_scores[np.nonzero(iou_mask)]

    return result




#載入權重
def load_weights(var_list, weights_file):

    with open(weights_file, "rb") as fp:
        _ = np.fromfile(fp, dtype=np.int32, count=5)#略過前5個int32
        weights = np.fromfile(fp, dtype=np.float32)

    ptr = 0
    i = 0
    assign_ops = []
    while i < len(var_list) - 1:
        var1 = var_list[i]
        var2 = var_list[i + 1]
        #找到卷冊積項
        if 'Conv' in var1.name.split('/')[-2]:
            # 找到BN參數項
            if 'BatchNorm' in var2.name.split('/')[-2]:
                # 載入批次歸一化參數
                gamma, beta, mean, var = var_list[i + 1:i + 5]
                batch_norm_vars = [beta, gamma, mean, var]
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(tf.assign(var, var_weights, validate_shape=True))

                i += 4#已經載入了4個變數，指標搬移4
            elif 'Conv' in var2.name.split('/')[-2]:
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
                ptr += bias_params
                assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))

                i += 1#搬移指標

            shape = var1.shape.as_list()
            num_params = np.prod(shape)
            #載入權重
            var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
            i += 1

    return assign_ops

#將等級結果顯示在圖片上
def draw_boxes(boxes, img, cls_names, detection_size):
    draw = ImageDraw.Draw(img)

    for cls, bboxs in boxes.items():
        color = tuple(np.random.randint(0, 256, 3))
        for box, score in bboxs:
            box = convert_to_original_size(box, np.array(detection_size), np.array(img.size))
            draw.rectangle(box, outline=color)
            draw.text(box[:2], '{} {:.2f}%'.format(cls_names[cls], score * 100), fill=color)
            print('{} {:.2f}%'.format(cls_names[cls], score * 100),box[:2])

def convert_to_original_size(box, size, original_size):
    ratio = original_size / size
    box = box.reshape(2, 2) * ratio
    return list(box.reshape(-1))


#載入資料集標簽名稱
def load_coco_names(file_name):
    names = {}
    with open(file_name) as f:
        for id, name in enumerate(f):
            names[id] = name
    return names

def main(argv=None):
    tf.reset_default_graph()
    img = Image.open(input_img)
    img_resized = img.resize(size=(size, size))

    classes = load_coco_names(class_names)

    #定義輸入占位符
    inputs = tf.placeholder(tf.float32, [None, size, size, 3])

    with tf.variable_scope('detector'):
        detections = yolo_v3(inputs, len(classes), data_format='NHWC')#定義網路結構
        #載入權重
        load_ops = load_weights(tf.global_variables(scope='detector'), weights_file)

    boxes = detections_boxes(detections)

    with tf.Session() as sess:
        sess.run(load_ops)

        detected_boxes = sess.run(boxes, feed_dict={inputs: [np.array(img_resized, dtype=np.float32)]})
    #對10647個預測框進行去重
    filtered_boxes = non_max_suppression(detected_boxes, confidence_threshold=conf_threshold,
                                         iou_threshold=iou_threshold)

    draw_boxes(filtered_boxes, img, classes, (size, size))

    img.save(output_img)
    img.show()


if __name__ == '__main__':
    main(_)
