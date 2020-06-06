# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室 
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""


import tensorflow as tf
model = __import__("5-2  model")
MyNASNetModel = model.MyNASNetModel

batch_size = 32
train_dir  = 'data/train'
val_dir  = 'data/val'

learning_rate1 = 1e-1
learning_rate2 = 1e-3

mymode = MyNASNetModel(r'nasnet-a_mobile_04_10_2017\model.ckpt')  #起始化模型
mymode.build_model('train',val_dir,train_dir,batch_size,learning_rate1 ,learning_rate2 )#將模型定義載入圖中

num_epochs1 = 2   #微調的迭代次數
num_epochs2 = 200#聯調的迭代次數

with tf.Session() as sess:
    sess.run(mymode.global_init)
   
    step = 0
    step = mymode.load_cpk(mymode.global_step,sess,1,mymode.saver,mymode.save_path )#載入模型
    print(step)
    if step == 0:#微調
        mymode.init_fn(sess)  #載入預先編譯模型權重
        
        for epoch in range(num_epochs1):
      		     
            print('Starting1 epoch %d / %d' % (epoch + 1, num_epochs1))#輸出進度 
            #用訓練集起始化迭代器
            sess.run(mymode.train_init_op)#資料集從頭開始
            while True:
                try:
                    step += 1
                    #預測，合並圖，訓練
                    acc,accuracy_top_5, summary, _ = sess.run([mymode.accuracy, mymode.accuracy_top_5,mymode.merged,mymode.last_train_op])
                    
                    #mymode.train_writer.add_summary(summary, step)#寫入記錄檔
                    if step % 100 == 0:
                        print(f'step: {step} train1 accuracy: {acc},{accuracy_top_5}')
                except tf.errors.OutOfRangeError:#資料集指標在最後
                    print("train1:",epoch," ok")
                    mymode.saver.save(sess, mymode.save_path+"/mynasnet.cpkt",   global_step=mymode.global_step.eval())
                    break
    
        sess.run(mymode.step_init)#微調結束，計數器從0開始
    
    #整體訓練
    for epoch in range(num_epochs2):
        print('Starting2 epoch %d / %d' % (epoch + 1, num_epochs2))
        sess.run(mymode.train_init_op)
        while True:
            try:
                step += 1
                #預測，合並圖，訓練
                acc, summary, _ = sess.run([mymode.accuracy, mymode.merged, mymode.full_train_op])
                
                mymode.train_writer.add_summary(summary, step)#寫入記錄檔

                if step % 100 == 0:
                    print(f'step: {step} train2 accuracy: {acc}')
            except tf.errors.OutOfRangeError:
                print("train2:",epoch," ok")
                mymode.saver.save(sess, mymode.save_path+"/mynasnet.cpkt",   global_step=mymode.global_step.eval())
                break

