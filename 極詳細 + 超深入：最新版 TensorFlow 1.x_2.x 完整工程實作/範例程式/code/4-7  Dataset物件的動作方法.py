"""
@author: 程式碼醫生工作室 
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""




# dataset ops
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import *
import numpy as np


###############  range(*args)



'''
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset = dataset.range(5)
iterator = dataset.make_one_shot_iterator()			#從到到尾讀一次
one_element = iterator.get_next()					#從iterator裡取出一個元素
with tf.Session() as sess:	# 建立階段（session）

    for i in range(5):		#透過for循環列印所有的資料
        print(sess.run(one_element))				#呼叫sess.run讀出Tensor值
'''



###############  zip(datasets)



'''
dataset1 = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset2 = tf.data.Dataset.from_tensor_slices(np.array([-1.0, -2.0, -3.0, -4.0, -5.0]))
dataset = Dataset.zip((dataset1,dataset2))
iterator = dataset.make_one_shot_iterator()			#從到到尾讀一次
one_element = iterator.get_next()					#從iterator裡取出一個元素
with tf.Session() as sess:	# 建立階段（session）

    for i in range(5):		#透過for循環列印所有的資料
        print(sess.run(one_element))				#呼叫sess.run讀出Tensor值
'''



###############  concatenate(dataset)



'''
dataset1 = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset2 = tf.data.Dataset.from_tensor_slices(np.array([-1.0, -2.0, -3.0, -4.0, -5.0]))
dataset = dataset1.concatenate(dataset2)
iterator = dataset.make_one_shot_iterator()			#從到到尾讀一次
one_element = iterator.get_next()					#從iterator裡取出一個元素
with tf.Session() as sess:	# 建立階段（session）

    for i in range(10):		#透過for循環列印所有的資料
        print(sess.run(one_element))				#呼叫sess.run讀出Tensor值
'''



###############  repeat(count=None)



'''
dataset1 = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset = dataset1.repeat(2)
iterator = dataset.make_one_shot_iterator()			#從到到尾讀一次
one_element = iterator.get_next()					#從iterator裡取出一個元素
with tf.Session() as sess:	# 建立階段（session）

    for i in range(10):		#透過for循環列印所有的資料
        print(sess.run(one_element))				#呼叫sess.run讀出Tensor值
'''


###############  shuffle(buffer_size,seed=None,reshuffle_each_iteration=None)


'''
dataset1 = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset = dataset1.shuffle(1000)
iterator = dataset.make_one_shot_iterator()			#從到到尾讀一次
one_element = iterator.get_next()					#從iterator裡取出一個元素
with tf.Session() as sess:	# 建立階段（session）

    for i in range(5):		#透過for循環列印所有的資料
        print(sess.run(one_element))				#呼叫sess.run讀出Tensor值

'''




###############  batch(count=None)



'''
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset = dataset.batch(batch_size=2)
iterator = dataset.make_one_shot_iterator()			#從到到尾讀一次
one_element = iterator.get_next()					#從iterator裡取出一個元素
with tf.Session() as sess:	# 建立階段（session）
	while True:
	    for i in range(2):		#透過for循環列印所有的資料
	        print(sess.run(one_element))				#呼叫sess.run讀出Tensor值
'''

###############  padded_batch

'''
data1 = tf.data.Dataset.from_tensor_slices([[1, 2],[1,3]])
data1 = data1.padded_batch(2,padded_shapes=[4])
iterator = data1.make_initializable_iterator()
next_element = iterator.get_next()
init_op = iterator.initializer

with tf.Session() as sess2:
    print(sess2.run(init_op))
    print("batched data 1:",sess2.run(next_element))
'''

###############  flat_map(map_func)




'''
import numpy as np

##在記憶體中產生資料
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset = tf.data.Dataset.from_tensor_slices(np.array([[1,2,3],[4,5,6]]))

dataset = dataset.flat_map(lambda x: Dataset.from_tensors(x)) 			
iterator = dataset.make_one_shot_iterator()		#從到到尾讀一次
one_element = iterator.get_next()				#從iterator裡取出一個元素
with tf.Session() as sess:						#建立階段（session）
    for i in range(10):							#透過for循環列印所有的資料
        print(sess.run(one_element))			#呼叫sess.run讀出Tensor值
'''



######interleave(map_func,cycle_length,block_length=1)



'''
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset = dataset.interleave(lambda x: Dataset.from_tensors(x).repeat(3),
             cycle_length=2, block_length=2)			
iterator = dataset.make_one_shot_iterator()		#從到到尾讀一次
one_element = iterator.get_next()				#從iterator裡取出一個元素
with tf.Session() as sess:						#建立階段（session）
    for i in range(100):							#透過for循環列印所有的資料
        print(sess.run(one_element),end=' ')			#呼叫sess.run讀出Tensor值
'''

######filter(predicate)



'''
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
dataset = dataset.filter(lambda x: tf.less(x, 3))			
iterator = dataset.make_one_shot_iterator()		#從到到尾讀一次
one_element = iterator.get_next()				#從iterator裡取出一個元素
with tf.Session() as sess:						#建立階段（session）
    for i in range(100):							#透過for循環列印所有的資料
        print(sess.run(one_element),end=' ')			#呼叫sess.run讀出Tensor值

#過濾掉全為0的元素
dataset = tf.data.Dataset.from_tensor_slices([ [0, 0],[ 3.0, 4.0] ])
dataset = dataset.filter(lambda x: tf.greater(tf.reduce_sum(x), 0))		  #過濾掉全為0的元素	
iterator = dataset.make_one_shot_iterator()		#從到到尾讀一次
one_element = iterator.get_next()				#從iterator裡取出一個元素
with tf.Session() as sess:						#建立階段（session）
    for i in range(100):							#透過for循環列印所有的資料
        print(sess.run(one_element),end=' ')			#呼叫sess.run讀出Tensor值

#過濾掉中文字串(1)加入一個判斷列
dataset = tf.data.Dataset.from_tensor_slices([ "hello","niha好" ])

def _parse_data(line):
    def checkone(line):
        for ch in line:
            #print(line,ch)
            if ch<23 or ch>127:
                return False
        return True
    isokstr = tf.py_func( checkone, [line], tf.bool)
    #tf.cast(isokstr,tf.bool)[0]

    return line,isokstr#tf.cast(isokstr,tf.bool)[0]
dataset = dataset.map(_parse_data)

dataset = dataset.filter(lambda x,y: y)		  #過濾掉全為0的元素	
iterator = dataset.make_one_shot_iterator()		#從到到尾讀一次
one_element = iterator.get_next()				#從iterator裡取出一個元素
with tf.Session() as sess:						#建立階段（session）
    for i in range(100):							#透過for循環列印所有的資料
        print(sess.run(one_element),end=' ')			#呼叫sess.run讀出Tensor值

#過濾掉中文字串(2)簡單實現
dataset = tf.data.Dataset.from_tensor_slices([ "hello","niha好" ])

def myfilter(x):
    def checkone(line):
        for ch in line:
            #print(line,ch)
            if ch<23 or ch>127:
                return False
        return True
    isokstr = tf.py_func( checkone, [x], tf.bool)
    return isokstr
dataset = dataset.filter(myfilter)		  #過濾掉全為0的元素	
#dataset = dataset.filter(lambda x,y: y)		  #過濾掉全為0的元素	
iterator = dataset.make_one_shot_iterator()		#從到到尾讀一次
one_element = iterator.get_next()				#從iterator裡取出一個元素
with tf.Session() as sess:						#建立階段（session）
    for i in range(100):							#透過for循環列印所有的資料
        print(sess.run(one_element),end=' ')			#呼叫sess.run讀出Tensor值

'''
######apply(transformation_func)
'''
data1 = np.arange(50).astype(np.int64)
dataset = tf.data.Dataset.from_tensor_slices(data1)
#將資料集中偶數行與奇數行分開，以window_size為視窗大小，一次取window_size個偶數行和window_size個奇數行。在window_size中，以batch為批次進行分割。
dataset = dataset.apply((tf.contrib.data.group_by_window(key_func=lambda x: x%2, reduce_func=lambda _, els: els.batch(10), window_size=20)  ))

iterator = dataset.make_one_shot_iterator()		#從到到尾讀一次
one_element = iterator.get_next()				#從iterator裡取出一個元素
with tf.Session() as sess:						#建立階段（session）
    for i in range(100):							#透過for循環列印所有的資料
        print(sess.run(one_element),end=' ')			#呼叫sess.run讀出Tensor值
'''





