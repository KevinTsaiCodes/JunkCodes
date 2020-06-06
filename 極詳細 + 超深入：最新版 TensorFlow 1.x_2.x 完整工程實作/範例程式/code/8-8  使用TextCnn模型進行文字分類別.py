# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室 
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""
import tensorflow as tf
import os
import time
import datetime

predata = __import__("8-6  NLP文字預先處理")
mydataset = predata.mydataset
text_cnn = __import__("8-7  TextCnn模型")
TextCNN = text_cnn.TextCNN
    
def train():
    #指定樣本檔案
    positive_data_file ="./data/rt-polaritydata/rt-polarity.pos"
    negative_data_file = "./data/rt-polaritydata/rt-polarity.neg"
    #設定訓練參數
    num_steps = 2000            #定義訓練次數
    display_every=20            #定義訓練中的顯示間隔
    checkpoint_every=100        #定義訓練中儲存模型的間隔
    SaveFileName= "text_cnn_model" #定義儲存模型資料夾名稱
    #設定模型參數    
    num_classes =2          #設定模型分類別
    dropout_keep_prob =0.8  #定義dropout系數
    l2_reg_lambda=0.1       #定義正則化系數
    filter_sizes = "3,4,5"  #定義多通道卷冊積核
    num_filters =64         #定義每通道的輸出個數
    
    tf.reset_default_graph()#清理圖
    
    #預先處理產生字典及資料集
    data,vocab_processor,max_document_length =mydataset(positive_data_file,negative_data_file)
    iterator = data.make_one_shot_iterator()
    next_element = iterator.get_next()
    
    #定義TextCnn網路
    cnn = TextCNN(
        sequence_length=max_document_length,
        num_classes=num_classes,
        vocab_size=len(vocab_processor.vocabulary_),
        embedding_size=128,
        filter_sizes=list(map(int, filter_sizes.split(","))),
        num_filters=num_filters,
        l2_reg_lambda=l2_reg_lambda)
    #建構網路
    cnn.build_mode()

    #開啟session，準備訓練
    session_conf = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    with tf.Session(config=session_conf) as sess:
        sess.run(tf.global_variables_initializer())
        

        #準備輸出模型路徑
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, SaveFileName, timestamp))
        print("Writing to {}\n".format(out_dir))

        #準備輸出摘要路徑
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        

        #準備檢查點名稱
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        #定義儲存檢查點的saver
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        #儲存字典
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        def train_step(x_batch, y_batch):#訓練步驟
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [cnn.train_op, cnn.global_step, cnn.train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            train_summary_writer.add_summary(summaries, step)
            return (time_str, step, loss, accuracy)

        i = 0
        while  tf.train.global_step(sess, cnn.global_step) < num_steps:
            x_batch, y_batch = sess.run(next_element)
            i = i+1
            time_str, step, loss, accuracy =train_step(x_batch, y_batch)
            
            current_step = tf.train.global_step(sess, cnn.global_step)
            if current_step % display_every == 0:
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                
            if current_step % checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

def main(argv=None):
    train()#啟動訓練

if __name__ == '__main__':
    tf.app.run()