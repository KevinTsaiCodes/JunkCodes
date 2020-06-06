# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室 
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""


import tensorflow as tf

def read_data(file_queue):                          #csv檔案處理函數
    reader = tf.TextLineReader(skip_header_lines=1)  #tf.TextLineReader 可以每次讀取一行
    key, value = reader.read(file_queue)
    
    defaults = [[0], [0.], [0.], [0.], [0.], [0]]       #為每個字段設定初值
    cvscolumn = tf.decode_csv(value, defaults)           #tf.decode_csv對每一行進行解析
    
    featurecolumn = [i for i in cvscolumn[1:-1]]        #分離出列中的樣本資料列
    labelcolumn = cvscolumn[-1]                         #分離出列中的標簽資料列
    
    return tf.stack(featurecolumn), labelcolumn         #傳回結果

def create_pipeline(filename, batch_size, num_epochs=None): #建立佇列資料集函數
    #建立一個輸加入佇列列
    file_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
    
    feature, label = read_data(file_queue)              #載入資料和標簽
    
    min_after_dequeue = 1000                            #設定佇列中的最少資料條數（取完資料後，確保還是有1000條）
    capacity = min_after_dequeue + batch_size              #佇列的長度
    
    feature_batch, label_batch = tf.train.shuffle_batch(    #產生亂序的批次資料
        [feature, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue
    )

    return feature_batch, label_batch                   #傳回指定批次資料

#讀取訓練集
x_train_batch, y_train_batch = create_pipeline('iris_training.csv', 32, num_epochs=100)
#讀取測試集
x_test, y_test = create_pipeline('iris_test.csv', 32)

with tf.Session() as sess:

    init_op = tf.global_variables_initializer()                 #起始化
    local_init_op = tf.local_variables_initializer()            #起始化本機變數，沒有回顯示出錯
    sess.run(init_op)
    sess.run(local_init_op)

    coord = tf.train.Coordinator()                          #建立協調器
    threads = tf.train.start_queue_runners(coord=coord)    #開啟執行緒列隊

    try:
        while True:
            if coord.should_stop():
                break
            example, label = sess.run([x_train_batch, y_train_batch]) #植入訓練資料
            print ("訓練資料：",example) #列印資料
            print ("訓練標簽：",label) #列印標簽
    except tf.errors.OutOfRangeError:       #定義取完資料的例外處理
        print ('Done reading')
        example, label = sess.run([x_test, y_test]) #植入測試資料
        print ("測試資料：",example) #列印資料
        print ("測試標簽：",label) #列印標簽
    except KeyboardInterrupt:               #定義按ctrl+c鍵時，對應的例外處理
        print("程式終止...")
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()