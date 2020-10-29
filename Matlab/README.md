# Notes

#### 次方: ^
###### Ex. 2^5 = 32

#### 查變數資料型態: whos 變數名

![未命名](https://user-images.githubusercontent.com/53148219/93421652-8244cb80-f8e4-11ea-8de0-bd08998f5397.png)

#### 註解: %

#### 清除畫面: clc

#### 程式結尾都用「;」

#### 開根號: sqrt

#### 查指令: help(怎麼使用)、lookfor 簡易查詢
![未命名](https://user-images.githubusercontent.com/53148219/93426669-14ea6800-f8ef-11ea-840b-abcabccc8898.png)

## week 02
### 宣告
#### 向量vector(一維陣列)
#### 宣告用 [  ]

#### 陣列起始值是「1」

#### 以「小括號」做 陣列切割符號
  Ex. S = [1, 2, 3, 4, 5]
	
	S(1) = 1
	S(3) = 3
#### length(a) 向量大小
#### 註解用 %
#### 向量加總 sum(a)
#### 向量元素相乘 prod(a)
#### 向量元素最大值 max(a)
#### 向量元素最小值 min(a)
#### 向量排序(由小至大, 排序後結果s, 索引值陣列i) [s i]= sort(a)
#### 向量排序(由大至小) [s i]= sort(a, ‘descend’);
#### sin(a)/cos(a)/tan(a)/cot(a): a為徑度量
#### sind(a)/cosd(a)/tand(a)/cotd(a): a為角度
#### 圓周 = 2pi 約 6.2832徑度量= 360度
#### clc; 清除畫面
#### a= -2*pi:0.1:2*pi % a 從 -2pi 到 2pi 每次增加 0.1 個單位。Ex. -2pi, -2pi+0.1, -2pi+0.1+0.1, ...., 2pi
#### plot(X_axis,Y_axis) 做圖表
#### subplot(X_size,Y_size, place_position)
#### place_position = 1,2,3,4 【1:左上,2:右上,3:左下,4:右下】
#### clear all; 清除所有變數包含自訂函數或圖形區域
#### img1= imread('file_name'); 讀入影像並指定給影像陣列
#### subplot(2, 3, [1, 2, 3]); 在2*3圖形區域合併圖形區 1,2,3
#### imshow(im1); 顯示 img1 影像陣列代表的影像

## week 04
![1](https://user-images.githubusercontent.com/53148219/96079115-89142d80-0ee6-11eb-959e-09960ded1658.jpg)

#### M 陣列的第3~8的元素各加根號5之後值給定給 a1
#### a1 = M(3:8)+sqrt(5)
#### M2 = [1 2 3; 4 5 6; 7 8 9] % 宣告 二維陣列 M2
#### mean(mean(M2)) % 求出 M2 二維陣列元素的平均
#### mean(M2) % 求出 M2 二維陣列每個 column 元素的平均
#### [c, r] = size(M2) % 求出 M2 的維度(高,寬)
#### M2 % 秀出矩陣 M2
#### M3 = M2' % 將 矩陣 M2 向左旋轉 90 度 給 M3
![1](https://user-images.githubusercontent.com/53148219/96085947-21fd7580-0ef4-11eb-9cc9-0ba4c7049ec2.jpg)
#### M1 = [1 2 3 4; 5 6 7 8]
#### M2 = reshape(M1, 4, 2) % reshape 矩陣變形, reshape(要改的矩陣, column, row)
#### 記住 原本 column x row = 後來的 column x row
#### format rat % 把小數弄成分數
#### ~= % 不等於
#### 從視窗中讀檔案 【uiget('副檔名')】
![螢幕擷取畫面 2020-10-22 134718](https://user-images.githubusercontent.com/53148219/96829938-1d453e00-146d-11eb-89cf-9388130303b7.jpg)
#### 照片垂直反轉
![螢幕擷取畫面 2020-10-22 140610](https://user-images.githubusercontent.com/53148219/96831546-db69c700-146f-11eb-9741-ef6770297979.jpg)
#### n plots in one graph
![3](https://user-images.githubusercontent.com/53148219/97530651-8d177380-19ed-11eb-816f-21c8275e0733.jpg)
#### Function (函式名稱要與檔案名稱一樣)
#### 宣告: function 回傳值 = 函式名(參數1,參數2..,參數n) % 如果回傳值是超過1個，用 [  ]  框起來
