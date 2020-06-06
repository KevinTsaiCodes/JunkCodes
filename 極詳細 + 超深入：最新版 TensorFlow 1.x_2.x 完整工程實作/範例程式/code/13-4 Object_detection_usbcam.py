# 匯入軟體包
import os
import cv2
import numpy as np
import tensorflow as tf
import sys



# 若果目前檔案在object_detection資料夾下，那麼將上一層路徑加入到python搜尋路徑中
sys.path.append('..')

# 匯入工具套件
from utils import label_map_util
from utils import visualization_utils as vis_util
# 設定攝影機解析度
IM_WIDTH = 640    # 使用較小的解析度，可以得到較快的檢驗框率
IM_HEIGHT = 480   

# 使用的模型名字
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

# 取得目前工作目錄的路徑
CWD_PATH = os.getcwd()

# 得到 detect model 檔案的路徑
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# 得到 label map 檔案路徑
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')

# 定義目的檢驗器檢驗的目的種類別數
NUM_CLASSES = 90

# 載入 label map，並產生檢驗種類別的索引值，以至於，當模型的前嚮推理計算出預測種類別是‘5’，我們
# 能夠知道對應的是 ‘飛機’
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# 載入 Tensorflow model 到記憶體中
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# 定義目的檢驗器的輸入，輸出張量

# 輸入張量是一幅圖形
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# 輸出張量是檢驗框，分值以及種類別
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# 檢驗到的目的種類別數
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# 起始化框率
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX


# 起始化USB攝影機
camera = cv2.VideoCapture(0)
ret = camera.set(3,IM_WIDTH)
ret = camera.set(4,IM_HEIGHT)

while(True):

    t1 = cv2.getTickCount()

    # 取得一副圖形，並延伸維度成：[1, None, None, 3]
    ret, frame = camera.read()
    frame_expanded = np.expand_dims(frame, axis=0)

    # 執行前嚮檢驗
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # 畫出檢驗的結果
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.85)

	# 畫出框率
    cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)
    
    # 顯示圖形
    cv2.imshow('Object detector', frame)

    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc = 1/time1

    # 按 'q' 離開
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()

cv2.destroyAllWindows()
