"""
@author: 程式碼醫生工作室
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""

import os
import tensorflow as tf

from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt



def load_sample(sample_dir,shuffleflag = True):
    '''遞歸讀取檔案。只支援一級。傳回檔名、數值標簽、數值對應的標簽名'''
    print ('loading sample  dataset..')
    lfilenames = []
    labelsnames = []
    for (dirpath, dirnames, filenames) in os.walk(sample_dir):#遞歸檢查資料夾
        for filename in filenames:                            #檢查所有檔名
            #print(dirnames)
            filename_path = os.sep.join([dirpath, filename])
            lfilenames.append(filename_path)               #加入檔名
            labelsnames.append( dirpath.split('\\')[-1] )#加入檔名對應的標簽

    lab= list(sorted(set(labelsnames)))  #產生標簽名稱清單
    labdict=dict( zip( lab  ,list(range(len(lab)))  )) #產生字典

    labels = [labdict[i] for i in labelsnames]
    if shuffleflag == True:
        return shuffle(np.asarray( lfilenames),np.asarray( labels)),np.asarray(lab)
    else:
        return (np.asarray( lfilenames),np.asarray( labels)),np.asarray(lab)


def _distorted_image(image,size,ch=1,shuffleflag = False,cropflag  = False,
                     brightnessflag=False,contrastflag=False):    #定義函數，實現變化圖片
    distorted_image =tf.image.random_flip_left_right(image)

    if cropflag == True:                                                #隨機裁剪
        s = tf.random_uniform((1,2),int(size[0]*0.8),size[0],tf.int32)
        distorted_image = tf.random_crop(distorted_image, [s[0][0],s[0][0],ch])

    distorted_image = tf.image.random_flip_up_down(distorted_image)#上下隨機翻轉
    if brightnessflag == True:#隨機變化亮度
        distorted_image = tf.image.random_brightness(distorted_image,max_delta=10)
    if contrastflag == True:   #隨機變化比較度
        distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)
    if shuffleflag==True:
        distorted_image = tf.random_shuffle(distorted_image)#沿著第0維亂序
    return distorted_image

from skimage import transform
def _random_rotated30(image, label): #定義函數實現圖片隨機旋轉動作

    def _rotated(image):                #封裝好的skimage模組，來進行圖片旋轉30度
        shift_y, shift_x = np.array(image.shape[:2],np.float32) / 2.
        tf_rotate = transform.SimilarityTransform(rotation=np.deg2rad(30))
        tf_shift = transform.SimilarityTransform(translation=[-shift_x, -shift_y])
        tf_shift_inv = transform.SimilarityTransform(translation=[shift_x, shift_y])
        image_rotated = transform.warp(image, (tf_shift + (tf_rotate + tf_shift_inv)).inverse)
        return image_rotated

    def _rotatedwrap():
        image_rotated = tf.py_func( _rotated,[image],[tf.float64])   #呼叫第三方函數
        return tf.cast(image_rotated,tf.float32)[0]

    a = tf.random_uniform([1],0,2,tf.int32)#實現隨機功能
    image_decoded = tf.cond(tf.equal(tf.constant(0),a[0]),lambda: image,_rotatedwrap)

    return image_decoded, label



def dataset(directory,size,batchsize,random_rotated=False,shuffleflag = True):#定義函數，建立資料集
    """ parse  dataset."""
    (filenames,labels),_ =load_sample(directory,shuffleflag=False) #載入檔名稱與標簽
    #print(filenames,labels)
    def _parseone(filename, label):                         #解析一個圖片檔案
        """ Reading and handle  image"""
        image_string = tf.read_file(filename)         #讀取整個檔案
        image_decoded = tf.image.decode_image(image_string)
        image_decoded.set_shape([None, None, None])    # 必須有這句，不然下面會轉化失敗
        image_decoded = _distorted_image(image_decoded,size)#對圖片做扭曲變化
        image_decoded = tf.image.resize_images(image_decoded, size)  #變化尺寸
        image_decoded = _norm_image(image_decoded,size)#歸一化
        image_decoded = tf.to_float(image_decoded)
        label = tf.cast(  tf.reshape(label, [1,]) ,tf.int32  )#將label 轉為張量
        return image_decoded, label

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))#產生Dataset物件


    if shuffleflag == True:#亂序
        dataset = dataset.shuffle(10000)

    dataset = dataset.map(_parseone)   #有圖片內容的資料集

    if random_rotated == True:#旋轉
        dataset = dataset.map(_random_rotated30)

    dataset = dataset.batch(batchsize) #批次劃分資料集
    dataset = dataset.prefetch(1)

    return dataset

def showresult(subplot,title,thisimg):          #顯示單一圖片
    p =plt.subplot(subplot)
    p.axis('off')
    p.imshow(thisimg)
    p.set_title(title)

def showimg(index,label,img,ntop):   #顯示
    plt.figure(figsize=(20,10))     #定義顯示圖片的寬、高
    plt.axis('off')
    ntop = min(ntop,9)
    print(index)
    for i in range (ntop):
        showresult(100+10*ntop+1+i,label[i],img[i])
    plt.show()
##########################
def _norm_image(image,size,ch=1,flattenflag = False):    #定義函數，實現歸一化，並且拍平
    image_decoded = image/127.5-1#image/255.0
    if flattenflag==True:
        image_decoded = tf.reshape(image_decoded, [size[0]*size[1]*ch])
    return image_decoded

from tensorflow.python.keras.applications.resnet50 import ResNet50

size = [224,224]
batchsize = 10

sample_dir=r"./apple2orange/train"
testsample_dir = r"./apple2orange/test"

traindataset = dataset(sample_dir,size,batchsize)#訓練集
testdataset = dataset(testsample_dir,size,batchsize,shuffleflag = False)#測試集

print(traindataset.output_types)  #列印資料集的輸出訊息
print(traindataset.output_shapes)


def imgs_input_fn(dataset):
    iterator = dataset.make_one_shot_iterator()			#產生一個迭代器
    one_element = iterator.get_next()					#從iterator裡取出一個元素
    return one_element

next_batch_train = imgs_input_fn(traindataset)				#從traindataset裡取出一個元素
next_batch_test = imgs_input_fn(testdataset)				#從testdataset裡取出一個元素

with tf.Session() as sess:	# 建立階段（session）
    sess.run(tf.global_variables_initializer())  #起始化

    try:
        for step in np.arange(1):
            value = sess.run(next_batch_train)
            showimg(step,value[1],np.asarray( (value[0]+1)*127.5,np.uint8),10)       #顯示圖片


    except tf.errors.OutOfRangeError:           #捕捉例外
        print("Done!!!")


###########################################
#建構模型
img_size = (224, 224, 3)
inputs = tf.keras.Input(shape=img_size)
conv_base = ResNet50(weights='resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',input_tensor=inputs,input_shape = img_size
                 ,include_top=False)#建立ResNet網路

model = tf.keras.models.Sequential()
model.add(conv_base)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
conv_base.trainable = False
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])




#訓練模型
model_dir ="./models/app2org"
os.makedirs(model_dir, exist_ok=True)
print("model_dir: ",model_dir)
est_app2org = tf.keras.estimator.model_to_estimator(keras_model=model,  model_dir=model_dir)

#訓練模型
train_spec = tf.estimator.TrainSpec(input_fn=lambda: imgs_input_fn(traindataset),
                                   max_steps=500)
eval_spec = tf.estimator.EvalSpec(input_fn=lambda: imgs_input_fn(testdataset))

import time
start_time = time.time()
tf.estimator.train_and_evaluate(est_app2org, train_spec, eval_spec)
print("--- %s seconds ---" % (time.time() - start_time))


#測試模型
img = value[0]
lab = value[1]

pre_input_fn = tf.estimator.inputs.numpy_input_fn(img,batch_size=10,shuffle=False)
predict_results = est_app2org.predict( input_fn=pre_input_fn)

predict_logits = []
for prediction in predict_results:
    print(prediction)
    predict_logits.append(prediction['dense_1'][0])

predict_is_org = [int(np.round(logit)) for logit in predict_logits]
actual_is_org = [int(np.round(label[0]))  for label in lab]
showimg(step,value[1],np.asarray( (value[0]+1)*127.5,np.uint8),10)
print("Predict :",predict_is_org)
print("Actual  :",actual_is_org)



