# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室 
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""
import tensorflow as tf
import time
import os
import numpy as np

import imageio 

Capsulemodel = __import__("8-2  Capsulemodel")
CapsuleNetModel = Capsulemodel.CapsuleNetModel

#載入資料集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./fashion/", one_hot=True)


#from official.mnist import mnist


def save_images(imgs, size, path):#定義函數儲存圖片
    imgs = (imgs + 1.) / 2  
    return(imageio.imwrite(path, mergeImgs(imgs, size)))
    

def mergeImgs(images, size):#定義函數，合並圖片
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image
    return imgs

batch_size = 128
learning_rate = 1e-3
training_epochs  = 5  #資料集迭代次數
n_class = 10

iter_routing = 3 #定義膠囊網路中動態路由的訓練次數

def main(_):
    
    capsmodel = CapsuleNetModel(batch_size, n_class,iter_routing) #案例化模型
    capsmodel.build_model(is_train=True,learning_rate=learning_rate)#建構網路節點

    os.makedirs('results', exist_ok=True)#建立路徑
    os.makedirs('./model', exist_ok=True)
        
    with tf.Session() as sess:  #建立階段
        sess.run(tf.global_variables_initializer())
        
        #載入檢查點
        checkpoint_path = tf.train.latest_checkpoint('./model/')
        print("checkpoint_path",checkpoint_path)
        if checkpoint_path !=None:
            capsmodel.saver.restore(sess, checkpoint_path)
        history = []
        for epoch in range(training_epochs):#按照指定次數迭代資料集

            total_batch = int(mnist.train.num_examples/batch_size)
            lossvalue= 0
            for i in range(total_batch):  #檢查資料集
                batch_x, batch_y = mnist.train.next_batch(batch_size)#取出資料
                batch_x = np.reshape(batch_x,[batch_size, 28, 28, 1])
                batch_y = np.asarray(batch_y,dtype=np.float32)
                
                tic = time.time()  #計算執行時間
                _, loss_value = sess.run([capsmodel.train_op, capsmodel.total_loss], feed_dict={capsmodel.x: batch_x,  capsmodel.y: batch_y})
                lossvalue +=loss_value
                if i % 20 == 0:#每訓練20次，輸出一次結果
                    print(str(i)+'用時：'+str(time.time()-tic)+' loss：',loss_value)
                    cls_result, recon_imgs = sess.run( [capsmodel.v_len, capsmodel.output], 
                                                      feed_dict={capsmodel.x: batch_x,  capsmodel.y: batch_y})
                    imgs = np.reshape(recon_imgs, (batch_size, 28, 28, 1))
                    size = 6
                    save_images(imgs[0:size * size, :], [size, size], 'results/test_%03d.png' % i)#將結果儲存為圖片

                    #獲得分類別結果，評估準確率
                    argmax_idx = np.argmax(cls_result,axis= 1)
                    batch_y_idx = np.argmax(batch_y,axis= 1)
                    print(argmax_idx[:3],batch_y_idx[:3])
                    cls_acc = np.mean(np.equal(argmax_idx, batch_y_idx).astype(np.float32))
                    print('正確率 : ' + str(cls_acc * 100)) 
            history.append(lossvalue/total_batch)
            if lossvalue/total_batch == min(history):
                ckpt_path = os.path.join('./model', 'model.ckpt')
                capsmodel.saver.save(sess, ckpt_path, global_step=capsmodel.global_step.eval())#儲存檢查點
                print("save model",ckpt_path)
            print(epoch,lossvalue/total_batch) 

if __name__ == "__main__":
    tf.app.run()