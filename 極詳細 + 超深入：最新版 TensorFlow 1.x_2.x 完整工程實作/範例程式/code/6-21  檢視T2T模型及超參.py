# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室 
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""

#6-19  


import tensorflow as tf
from tensor2tensor import models

from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import registry

print(len(registry.list_models()),registry.list_models())
print(registry.model('transformer'))
print(len(registry.list_hparams()),registry.list_hparams('transformer'))
print(registry.hparams('transformer_base_v1'))