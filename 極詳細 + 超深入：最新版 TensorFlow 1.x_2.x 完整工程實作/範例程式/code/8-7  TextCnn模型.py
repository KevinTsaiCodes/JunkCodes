"""
@author: 程式碼醫生工作室 
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

class TextCNN(object):
    """
    TextCNN文字分類別器.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        
        #定義占位符
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        #詞內嵌層 
        with tf.variable_scope('Embedding'):
            embed = tf.contrib.layers.embed_sequence(self.input_x, vocab_size=vocab_size, embed_dim=embedding_size)
            self.embedded_chars_expanded = tf.expand_dims(embed, -1)

        #定義多通道卷冊積 與最大池化網路
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            conv = slim.conv2d(self.embedded_chars_expanded, num_outputs = num_filters, 
                            kernel_size=[filter_size,embedding_size], 
                            stride=1, padding="VALID",
                            activation_fn=tf.nn.leaky_relu,scope="conv%s" % filter_size)
            pooled = slim.max_pool2d(conv, [sequence_length - filter_size + 1, 1], padding='VALID',
                scope="pool%s" % filter_size)

            pooled_outputs.append(pooled)

        #展開特征，並加入dropout
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        #計算L2_loss
        l2_loss = tf.constant(0.0)
        with tf.name_scope("output"):
            self.scores = slim.fully_connected(self.h_drop, num_classes, activation_fn=None,scope="fully_connected" )
            for tf_var in tf.trainable_variables():
                if ("fully_connected" in tf_var.name ):
                    l2_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))
                    print("tf_var",tf_var)

            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # 計算交叉熵
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        #計算準確率
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
   
    def build_mode(self):#定義函數建構模型
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        #產生摘要
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)
        #產生損失及準確率的摘要
        loss_summary = tf.summary.scalar("loss", self.loss)
        acc_summary = tf.summary.scalar("accuracy", self.accuracy)

        #合並摘要
        self.train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])

        