# -*- coding: utf-8 -*-
"""
@author: 程式碼醫生工作室 
@公眾號：xiangyuejiqiren   （內有更多優秀文章及研讀資料）
@來源: <深度研讀之TensorFlow專案化專案實戰>配套程式碼 （700+頁）
@配套程式碼技術支援：bbs.aianaconda.com      (有問必答)
"""
import grpc
import numpy as np
import tensorflow as tf
import time
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

def client_gRPC(data):

    channel = grpc.insecure_channel('127.0.0.1:9000')#建立一個通道
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)#連結遠端伺服器
     
    #起始化請求
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'md' #指定模型名稱
    request.model_spec.signature_name = "my_signature" #指定模型簽名
    request.inputs['input_x'].CopyFrom(tf.contrib.util.make_tensor_proto(data))
    
    #開始呼叫遠端服務。執行預測工作
    start_time = time.time()
    result = stub.Predict(request)
    
    #輸出預測時間
    print("花費時間: {}".format(time.time()-start_time))
    
    #解析結果並傳回
    result_dict = {}
    for key in result.outputs:
        tensor_proto = result.outputs[key]
        nd_array = tf.contrib.util.make_ndarray(tensor_proto)
        result_dict[key] = nd_array

    return result_dict

def main():
    a = 4.2#傳入單一數值
    result= client_gRPC(a)
    print("-------單一數值預測結果-------")
    print(list(result['output']))
    
    #傳入多個數值
    data = np.asarray([4.2,4.0],dtype = np.float32)  
    result= client_gRPC(data)
    print("-------多個數值預測結果-------")
    print(list(result['output']))

#主模組執行函數
if __name__ == '__main__':
        main()

