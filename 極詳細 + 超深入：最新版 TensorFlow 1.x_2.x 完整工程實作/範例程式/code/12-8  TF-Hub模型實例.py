# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室 
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""

import os
import shutil
import tempfile
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


#定義詞內嵌內容
_MOCK_EMBEDDING = "\n".join(
    ["cat 1.11 2.56 3.45", "dog 1 2 3", "mouse 0.5 0.1 0.6"])
_embedding_file_path  = "./mock_embedding_file.txt"
#產生詞內嵌檔案
with tf.gfile.GFile(_embedding_file_path, mode="w") as f:
      f.write(_MOCK_EMBEDDING)



def parse_line(line):#解析詞內嵌檔案中的一行
  columns = line.split()
  token = columns.pop(0)
  values = [float(column) for column in columns]
  return token, values

def load(file_path, parse_line_fn):#按照特殊的方法，載入詞內嵌
  vocabulary = []
  embeddings = []
  embeddings_dim = None
  for line in tf.gfile.GFile(file_path):
    token, embedding = parse_line_fn(line)
    if not embeddings_dim:
      embeddings_dim = len(embedding)
    elif embeddings_dim != len(embedding):
      raise ValueError(
          "Inconsistent embedding dimension detected, %d != %d for token %s",
          embeddings_dim, len(embedding), token)
    vocabulary.append(token)
    embeddings.append(embedding)
  return vocabulary, np.array(embeddings)

def make_module_spec(vocabulary_file, vocab_size, embeddings_dim, #傳回TF-Hub的spec模型
                     num_oov_buckets, preprocess_text):
  def module_fn():#標準的不帶預先處理模型
    tokens = tf.placeholder(shape=[None], dtype=tf.string, name="tokens")
    embeddings_var = tf.get_variable(#定義詞內嵌變數
        initializer=tf.zeros([vocab_size + num_oov_buckets, embeddings_dim]),
        name='embedding',dtype=tf.float32)

    lookup_table = tf.contrib.lookup.index_table_from_file(
        vocabulary_file=vocabulary_file,
        num_oov_buckets=num_oov_buckets)

    ids = lookup_table.lookup(tokens)
    combined_embedding = tf.nn.embedding_lookup(params=embeddings_var, ids=ids)
    hub.add_signature("default", {"tokens": tokens},
                      {"default": combined_embedding})

  def module_fn_with_preprocessing():#支援全文字輸入，帶有預先處理的模型
    sentences = tf.placeholder(shape=[None], dtype=tf.string, name="sentences")

    #使用正規表示法，移除特殊符號
    normalized_sentences = tf.regex_replace(
        input=sentences, pattern=r"\pP", rewrite="")
    #按照空格分詞，得到稀疏矩陣
    tokens = tf.string_split(normalized_sentences, " ")

    embeddings_var = tf.get_variable(#定義詞內嵌變數
        initializer=tf.zeros([vocab_size + num_oov_buckets, embeddings_dim]),
        name='embedding', dtype=tf.float32)

    #用字典將詞變為詞向量
    lookup_table = tf.contrib.lookup.index_table_from_file(
        vocabulary_file=vocabulary_file,
        num_oov_buckets=num_oov_buckets)

    #將稀疏矩陣用詞內嵌轉化
    sparse_ids = tf.SparseTensor(
        indices=tokens.indices,
        values=lookup_table.lookup(tokens.values),
        dense_shape=tokens.dense_shape)

    #為稀疏矩陣加入空行
    sparse_ids, _ = tf.sparse_fill_empty_rows(
        sparse_ids, lookup_table.lookup(tf.constant("")))

    #sparse_ids = tf.sparse_reset_shape(sparse_ids)
    #結果進行平方和再開根號的歸約計算
    combined_embedding = tf.nn.embedding_lookup_sparse(
        params=embeddings_var,sp_ids=sparse_ids,
        sp_weights=None, combiner="sqrtn")


    #預設都統一使用default簽名。若果額外指定，還需要在呼叫時與其對應
    #輸入和輸出需要字典形式。可以是多個
    hub.add_signature("default", {"sentences": sentences},
                      {"default": combined_embedding})

  if preprocess_text:
    return hub.create_module_spec(module_fn_with_preprocessing)
  else:
    return hub.create_module_spec(module_fn)

#匯出TF-Hub模型
def export(export_path, vocabulary, embeddings, num_oov_buckets,
           preprocess_text):#模型是否支援預先處理

  #建立暫存檔夾
  tmpdir = tempfile.mkdtemp()
  #建立目錄
  vocabulary_file = os.path.join(tmpdir, "tokens.txt")

  #將字典vocabulary寫入檔案
  with tf.gfile.GFile(vocabulary_file, "w") as f:
    f.write("\n".join(vocabulary))

  spec = make_module_spec(vocabulary_file, len(vocabulary), embeddings.shape[1],
                          num_oov_buckets, preprocess_text)
  try:
    with tf.Graph().as_default():
        
      m = hub.Module(spec) 
      p_embeddings = tf.placeholder(tf.float32)
      #為定義好的詞內嵌給予值（還原模型）
      load_embeddings = tf.assign(m.variable_map['embedding'],
                                  p_embeddings)

      with tf.Session() as sess:
        #以植入的模式將模型權重還原到模型中去
        sess.run([load_embeddings], feed_dict={p_embeddings: embeddings})
        m.export(export_path, sess)#產生模型

  finally:
    shutil.rmtree(tmpdir)

#按照num_oov_buckets個數，擴充詞內嵌，支援轉化時的不詳字元。
def maybe_append_oov_vectors(embeddings, num_oov_buckets):
  num_embeddings = np.shape(embeddings)[0]
  embedding_dim = np.shape(embeddings)[1]
  embeddings.resize(
      [num_embeddings + num_oov_buckets, embedding_dim], refcheck=False)


def export_module_from_file(embedding_file, export_path, parse_line_fn,
                            num_oov_buckets, preprocess_text):
  #載入詞內嵌檔案到記憶體
  vocabulary, embeddings = load(embedding_file, parse_line_fn)

  #按照num_oov_buckets個數，擴充詞內嵌，支援轉化時的不詳字元。
  maybe_append_oov_vectors(embeddings, num_oov_buckets)

  #將模型匯出
  export(export_path, vocabulary, embeddings, num_oov_buckets, preprocess_text)


###使用

 
os.makedirs('./emb', exist_ok=True)  
os.makedirs('./peremb', exist_ok=True)      
#產生一個詞內嵌模型
export_module_from_file(
        embedding_file=_embedding_file_path,
        export_path='./emb',
        parse_line_fn=parse_line,
        num_oov_buckets=1,
        preprocess_text=False)

#產生一個帶有預先處理的詞內嵌模型
export_module_from_file(
        embedding_file=_embedding_file_path,
        export_path='./peremb',
        parse_line_fn=parse_line,
        num_oov_buckets=1,
        preprocess_text=True)

with tf.Graph().as_default():
      hub_module = hub.Module('./emb')
      tokens = tf.constant(["cat", "lizard", "dog"])
      
      perhub_module = hub.Module('./peremb')
      pertesttokens = tf.constant(["cat", "cat cat", "lizard. dog", "cat? dog", ""])

      embeddings = hub_module(tokens)
      perembeddings = perhub_module(pertesttokens)
      with tf.Session() as session:

        session.run(tf.tables_initializer())
        session.run(tf.global_variables_initializer())
        print(session.run(embeddings))
        print(session.run(perembeddings))












#
#
#import tensorflow as tf
#
#sentences = tf.placeholder(shape=[None], dtype=tf.string, name="sentences")
##5個句子
#tokensiniput = tf.constant(["cat", "cat cat", "lizard. dog", "cat? dog", ""])
##去掉其它符號
#normalized_sentences = tf.regex_replace( input=tokensiniput, pattern=r"\pP", rewrite="")
##自動對齊，補0
#tokens = tf.string_split(normalized_sentences, " ")
#tokens2, _ = tf.sparse_fill_empty_rows(
#        tokens, tf.constant(""))
#
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    print(sess.run(normalized_sentences))
#    print(sess.run(tokens))    
#    print(sess.run(tokens2))