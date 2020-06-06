# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""
import sys
nets_path = r'slim'                             #載入環境變數
if nets_path not in sys.path:
    sys.path.insert(0,nets_path)
else:
    print('already add slim')

import tensorflow as tf
from nets.nasnet import nasnet                 #匯出nasnet
slim = tf.contrib.slim

import os
mydataset = __import__("5-1  mydataset")
creat_dataset_fromdir = mydataset.creat_dataset_fromdir

class MyNASNetModel(object):
    """微調模型類別MyNASNetModel
    """
    def __init__(self, model_path=''):
        self.model_path = model_path  #原始模型的路徑

    def MyNASNet(self,images,is_training):
        arg_scope = nasnet.nasnet_mobile_arg_scope()          #獲得模型命名空間
        with slim.arg_scope(arg_scope):
            #建構NASNet Mobile模型
            logits, end_points = nasnet.build_nasnet_mobile(images,num_classes = self.num_classes+1,
                                                            is_training=is_training)

        global_step = tf.train.get_or_create_global_step()  #定義記錄步數的張量

        return logits,end_points,global_step   #傳回有用的張量

    def FineTuneNASNet(self,is_training):       #實現微調模型的網路動作
        model_path = self.model_path

        exclude = ['final_layer','aux_7']  #還原超參， 除了exclude以外的全部還原
        variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
        if is_training == True:
            init_fn = slim.assign_from_checkpoint_fn(model_path, variables_to_restore,ignore_missing_vars=True)
        else:
            init_fn = None

        tuning_variables = [] #將沒有還原的超參收集起來，用於微調訓練
        for v in exclude:
            tuning_variables += slim.get_variables(v)

        print('final_layer:',slim.get_variables('final_layer'))
        print('aux_7:',slim.get_variables('aux_7'))
        print("tuning_variables:",tuning_variables)

        return init_fn,tuning_variables



    def build_acc_base(self,labels):#定義評估函數
        #傳回張量中最大值的索引
        self.prediction = tf.cast(tf.argmax(self.logits, 1),tf.int32)
        #計算prediction、labels是否相同
        self.correct_prediction = tf.equal(self.prediction, labels)
        #計算平均值
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,tf.float32))
        #計算這些目的是否在最高的前5預測中，並取平均值
        self.accuracy_top_5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=self.logits, targets=labels, k=5),tf.float32))

    def load_cpk(self,global_step,sess,begin = 0,saver= None,save_path = None):
        """
        模型儲存和匯出
        """
        if begin == 0:
            save_path=r'./train_nasnet'  #定義檢查點路徑
            if not os.path.exists(save_path):
                print("there is not a model path:",save_path)
            saver = tf.train.Saver(max_to_keep=1) # 產生saver
            return saver,save_path
        else:
            kpt = tf.train.latest_checkpoint(save_path)#查詢最新的檢查點
            print("load model:",kpt)
            startepo= 0#計步
            if kpt!=None:
                saver.restore(sess, kpt) #復原模型
                ind = kpt.find("-")
                startepo = int(kpt[ind+1:])
                print("global_step=",global_step.eval(),startepo)
            return startepo

    def build_model_train(self,images, labels,learning_rate1,learning_rate2,is_training):
        self.logits,self.end_points, self.global_step= self.MyNASNet(images,is_training=is_training)
        self.step_init = self.global_step.initializer
        self.init_fn,self.tuning_variables = self.FineTuneNASNet(is_training=is_training)
        #定義損失函數
        tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=self.logits)
        loss = tf.losses.get_total_loss()

        #定義退化研讀速率
        learning_rate1 = tf.train.exponential_decay(learning_rate=learning_rate1,#微調的研讀率
                global_step=self.global_step,
                decay_steps=100, decay_rate=0.5)
        learning_rate2 = tf.train.exponential_decay(learning_rate=learning_rate2,#聯調的研讀率
                global_step=self.global_step,
                decay_steps=100, decay_rate=0.2)

        #定義沖量Momentum改善器
#        last_optimizer = tf.train.MomentumOptimizer(learning_rate1, 0.8, use_nesterov=True)
#        full_optimizer = tf.train.MomentumOptimizer(learning_rate2, 0.8, use_nesterov=True)

        last_optimizer = tf.train.AdamOptimizer(learning_rate1)
        full_optimizer = tf.train.AdamOptimizer(learning_rate2)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):#更新批次歸一化中的參數
            #使loss減小方向做改善
            self.last_train_op = last_optimizer.minimize(loss, self.global_step,var_list=self.tuning_variables)
            self.full_train_op = full_optimizer.minimize(loss, self.global_step)

        #self.opt_init = [var.initializer for var in tf.global_variables() if 'Momentum' in var.name]
#        self.opt_init = [var.initializer for var in tf.global_variables() if 'Adam' in var.name]


        self.build_acc_base(labels)#定義評估模型關聯指標

        tf.summary.scalar('accuracy', self.accuracy)#寫入日志，支援tensorBoard動作
        tf.summary.scalar('accuracy_top_5', self.accuracy_top_5)

        #將收集的所有預設圖表並合並
        self.merged = tf.summary.merge_all()
        #寫入記錄檔
        self.train_writer = tf.summary.FileWriter('./log_dir/train')
        self.eval_writer = tf.summary.FileWriter('./log_dir/eval')

        self.saver,self.save_path = self.load_cpk(self.global_step,None)   #定義檢查點關聯變數



    def build_model(self,mode='train',testdata_dir='./data/val',traindata_dir='./data/train', batch_size=32,learning_rate1=0.001,learning_rate2=0.001):

        if mode == 'train':
            tf.reset_default_graph()
            #建立訓練資料和測試資料的Dataset資料集
            dataset,self.num_classes = creat_dataset_fromdir(traindata_dir,batch_size)
            testdataset,_ = creat_dataset_fromdir(testdata_dir,batch_size,isTrain = False)

            #建立一個可起始化的迭代器
            iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
            #讀取資料
            images, labels = iterator.get_next()
            iterator.make_initializer

            self.train_init_op = iterator.make_initializer(dataset)
            self.test_init_op = iterator.make_initializer(testdataset)

            self.build_model_train(images, labels,learning_rate1,learning_rate2,is_training=True)

            self.global_init = tf.global_variables_initializer()
            tf.get_default_graph().finalize()


        elif mode == 'test':
            tf.reset_default_graph()

            #建立測試資料的Dataset資料集
            testdataset,self.num_classes = creat_dataset_fromdir(testdata_dir,batch_size,isTrain = False)

            #建立一個可起始化的迭代器
            iterator = tf.data.Iterator.from_structure(testdataset.output_types, testdataset.output_shapes)
            #讀取資料
            self.images, labels = iterator.get_next()

            self.test_init_op = iterator.make_initializer(testdataset)
            self.logits,self.end_points, self.global_step= self.MyNASNet(self.images,is_training=False)
            self.saver,self.save_path = self.load_cpk(self.global_step,None)   #定義檢查點關聯變數
            #評估指標
            self.build_acc_base(labels)
            tf.get_default_graph().finalize()


        elif mode == 'eval':
            tf.reset_default_graph()
            #建立測試資料的Dataset資料集
            testdataset,self.num_classes = creat_dataset_fromdir(testdata_dir,batch_size,isTrain = False)

            #建立一個可起始化的迭代器
            iterator = tf.data.Iterator.from_structure(testdataset.output_types, testdataset.output_shapes)
            #讀取資料
            self.images, labels = iterator.get_next()


            self.logits,self.end_points, self.global_step= self.MyNASNet(self.images,is_training=False)
            self.saver,self.save_path = self.load_cpk(self.global_step,None)   #定義檢查點關聯變數
            tf.get_default_graph().finalize()
