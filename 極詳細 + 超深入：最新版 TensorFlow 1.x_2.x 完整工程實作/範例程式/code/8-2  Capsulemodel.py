# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室 
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class CapsuleNetModel:  #定義膠囊網路模型類別
    def __init__(self, batch_size,n_classes,iter_routing):#起始化
        self.batch_size=batch_size
        self.n_classes = n_classes
        self.iter_routing = iter_routing

    def CapsuleNet(self, img):#定義網路模型結構
        
        with tf.variable_scope('Conv1_layer') as scope:#定義第一個標準卷冊積層 ReLU Conv1
            output = slim.conv2d(img, num_outputs=256, kernel_size=[9, 9], stride=1, padding='VALID', scope=scope)
            assert output.get_shape() == [self.batch_size, 20, 20, 256]
    
        with tf.variable_scope('PrimaryCaps_layer') as scope:#定義主膠囊網路 Primary Caps
            output = slim.conv2d(output, num_outputs=32*8, kernel_size=[9, 9], stride=2, padding='VALID', scope=scope, activation_fn=None)
            output = tf.reshape(output, [self.batch_size, -1, 1, 8])  #將結果變成32*6*6個膠囊單元，每個單元為8維向量
            assert output.get_shape() == [self.batch_size, 1152, 1, 8]

        with tf.variable_scope('DigitCaps_layer') as scope:#定義數字膠囊 Digit Caps
            u_hats = []
            input_groups = tf.split(axis=1, num_or_size_splits=1152, value=output)#將輸入按照膠囊單元分開
            for i in range(1152): #檢查每個膠囊單元
                #利用卷冊積核為[1，1]的卷冊積動作，讓u與w相乘，再相加得到u_hat
                one_u_hat = slim.conv2d(input_groups[i], num_outputs=16*10, kernel_size=[1, 1], stride=1, padding='VALID', scope='DigitCaps_layer_w_'+str(i), activation_fn=None)
                one_u_hat = tf.reshape(one_u_hat, [self.batch_size, 1, 10, 16])#每個膠囊單元變成了16維向量
                u_hats.append(one_u_hat)
            
            u_hat = tf.concat(u_hats, axis=1)#將所有的膠囊單元中的one_u_hat合並起來
            assert u_hat.get_shape() == [self.batch_size, 1152, 10, 16]

            #起始化b值
            b_ijs = tf.constant(np.zeros([1152, 10], dtype=np.float32))
            v_js = []
            for r_iter in range(self.iter_routing):#按照指定循環次數，計算動態路由
                with tf.variable_scope('iter_'+str(r_iter)):
                    c_ijs = tf.nn.softmax(b_ijs, axis=1)  #根據b值，獲得耦合系數

                    #將下列變數按照10類別分割，每一類別單獨運算
                    c_ij_groups = tf.split(axis=1, num_or_size_splits=10, value=c_ijs)
                    b_ij_groups = tf.split(axis=1, num_or_size_splits=10, value=b_ijs)
                    u_hat_groups = tf.split(axis=2, num_or_size_splits=10, value=u_hat)

                    for i in range(10):
                        #產生具有跟輸入一樣尺寸的卷冊積核[1152, 1]，輸入為16通道,卷冊積核個數為1個
                        c_ij = tf.reshape(tf.tile(c_ij_groups[i], [1, 16]), [1152, 1, 16, 1])
                        #利用深度卷冊積實現u_hat與c矩陣的對應位置相乘，輸出的通道數為16*1個
                        s_j = tf.nn.depthwise_conv2d(u_hat_groups[i], c_ij, strides=[1, 1, 1, 1], padding='VALID')
                        assert s_j.get_shape() == [self.batch_size, 1, 1, 16]

                        s_j = tf.reshape(s_j, [self.batch_size, 16])
                        v_j = self.squash(s_j)  #使用squash啟動函數，產生最終的輸出vj
                        assert v_j.get_shape() == [self.batch_size, 16]
                        #根據vj來計算，並更新b值
                        b_ij_groups[i] = b_ij_groups[i]+tf.reduce_sum(tf.matmul(tf.reshape(u_hat_groups[i], 
                                   [self.batch_size, 1152, 16]), tf.reshape(v_j, [self.batch_size, 16, 1])), axis=0)

                        if r_iter == self.iter_routing-1:  #迭代結束後，再產生一次vj，得到數字膠囊真正的輸出結果
                            v_js.append(tf.reshape(v_j, [self.batch_size, 1, 16]))

                    b_ijs = tf.concat(b_ij_groups, axis=1)#將10類別的b合並到一起

            output = tf.concat(v_js, axis=1)#將10類別的vj合並到一起，產生的形狀為[self.batch_size, 10, 16]的結果

        return  output
    def squash(self, s_j):  #定義啟動函數
        s_j_norm_square = tf.reduce_mean(tf.square(s_j), axis=1, keepdims=True)
        v_j = s_j_norm_square*s_j/((1+s_j_norm_square)*tf.sqrt(s_j_norm_square+1e-9))
        return v_j
    
    def build_model(self, is_train=False,learning_rate = 1e-3):
        tf.reset_default_graph()

        #定義占位符
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.n_classes])
        self.x = tf.placeholder(tf.float32, [self.batch_size, 28, 28, 1], name='input')
        
        #定義計步器
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        
        
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        biasInitializer = tf.constant_initializer(0.0)
    
        with slim.arg_scope([slim.conv2d], trainable=is_train, weights_initializer=initializer, biases_initializer=biasInitializer):
            self.v_jsoutput = self.CapsuleNet(self.x) #建構膠囊網路
            
            tf.check_numerics(self.v_jsoutput,"self.v_jsoutput is nan ")#判斷張量是否為nan 或inf
            
            with tf.variable_scope('Masking'):  
                self.v_len = tf.norm(self.v_jsoutput, axis=2)#計算輸出值的歐幾裡得范數[self.batch_size, 10]
                
                
    
            if is_train:            #若果是訓練模式，重建輸入圖片
                masked_v = tf.matmul(self.v_jsoutput, tf.reshape(self.y, [-1, 10, 1]), transpose_a=True)
                masked_v = tf.reshape(masked_v, [-1, 16])
    
                with tf.variable_scope('Decoder'):
                    output = slim.fully_connected(masked_v, 512, trainable=is_train)
                    output = slim.fully_connected(output, 1024, trainable=is_train)
                    self.output = slim.fully_connected(output, 784, trainable=is_train, activation_fn=tf.sigmoid)
    
                self.total_loss = self.loss(self.v_len,self.output)#計算loss值
                #使用退化研讀率
                learning_rate_decay = tf.train.exponential_decay(learning_rate, global_step=self.global_step, decay_steps=2000,decay_rate=0.9)
    
                #定義改善器
                self.train_op = tf.train.AdamOptimizer(learning_rate_decay).minimize(self.total_loss, global_step=self.global_step)
                
        #定義儲存及還原模型關鍵點所使用的saver
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)


    def loss(self,v_len, output): #定義loss計算函數
        max_l = tf.square(tf.maximum(0., 0.9-v_len))
        max_r = tf.square(tf.maximum(0., v_len - 0.1))
        
        l_c = self.y*max_l+0.5 * (1 - self.y) * max_r
    
        margin_loss = tf.reduce_mean(tf.reduce_sum(l_c, axis=1))
    
        origin = tf.reshape(self.x, shape=[self.batch_size, -1])
        reconstruction_err = tf.reduce_mean(tf.square(output-origin))
    
        total_loss = margin_loss+0.0005*reconstruction_err#將邊距損失與重建損失一起構成loss

    
        return total_loss

