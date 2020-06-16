import tensorflow as tf
from tensorflow.keras.models import  Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Dropout,Conv2D,MaxPooling2D, Flatten
import pickle


X = pickle.load(open("X.pickle", "rb"))

y = pickle.load(open("Y.pickle", "rb"))

"""
Above has the meaning is same as below:
	pickle_in = open("X.pickle","rb")
	X = pickle.load(pickle_in)

	pickle_in = open("y.pickle","rb")
	y = pickle.load(pickle_in)
"""
X = X/255.0 # let image to 0~255

model = Sequential()
model.add(Conv2D(256,(3,3),input_shape=X.shape[1:]))
""" Conv2D(filter,kernel_size,strides)
	
	filters:
		the dimensionality of the output space
	kernel_size:
		An integer or tuple/list of 2 integers,
		specifying the height and width of the 2D convolution window.
		Can be a single integer to specify the same value for all spatial dimensions.
	strides:
		An integer or tuple/list of 2 integers, specifying the strides of the convolution
		along the height and width. 
"""
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) # pool_size: 型別是整數，最大池化窗口大小
"""
池化層
	除了卷積層，卷積網絡也經常使用池化層來縮減模型的大小，提高計算速度，
	同時提高所提取特徵的魯棒性，池化只是計算神經網絡某一層的靜態屬性。
"""
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64)) # Dense, 稠密度，
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
"""
損失函數(loss): 

當預測值與實際值愈相近，損失函數就愈小，反之差距很大，就會
更影響損失函數的值，這篇文章 主張要用Cross Entropy 取代 
MSE，因為，在梯度下時，Cross Entropy 計算速度較快，其他
變形包括 sparse_categorical_crossentropy

優化函數(optimizer): 

一般而言，比SGD模型訓練成本較低

成效衡量指標(mertrics)
"""
model.fit(X, y, batch_size=5, epochs=15, verbose=1)

model.summary()
"""
model.fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1)

x：
輸入數據。如果模型只有一個輸入，那麼x的類型是numpy array，如果模型有多個輸入，
那麼x的類型應當為list，list的元素是對應於各個輸入的numpy array。如果模型的每個輸
入都有名字，則可以傳入一個字典，將輸入名與其輸入數據對應起來。

y:
標籤，numpy array。如果模型有多個輸出，可以傳入一個numpy array的list。如果模型的輸
出擁有名字，則可以傳入一個字典，將輸出名與其標籤對應起來。

batch_size:
整數，指定進行梯度下降時每個batch包含的樣本數。訓練時一個batch的樣本會被計算一次梯度下降，使目標函數優化一步。

epochs:
整數，訓練終止時的epoch值，訓練將在達到該epoch值時停止

verbose:
日誌顯示，0為不在標準輸出流輸出日誌信息，1為輸出進度條記錄，2為每個epoch輸出一行記錄
"""
