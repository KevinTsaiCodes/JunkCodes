#include <stdio.h> 
#include <stdlib.h> 
int main()
{ //主程式
   int Temp[]={1,8,15,24,33,45,76,88,99};
   int Key;
   int Low, High, Middle, Searchtime;
   printf("===============程式描述================\n");
   printf("= 程式名稱：ch9-3.c                   =\n");
   printf("= 程式目的：二分法搜尋                =\n"); 
   printf("=======================================\n"); 
   printf("===========輸入(預先排序)==============\n");
   printf("1, 8, 15, 24, 33, 45, 76, 88, 99\n");
   printf("\n"); 
   printf("請輸入以上的任一組數字：");
   scanf("%d", &Key); 
   Low = 0;                                   //設定第一個字母
   High = 8;                                  //設定最後一個字母
   Searchtime = 0;                            //搜尋次數初值設定為
   Middle = (int)((Low + High)/2);            //搜尋中間值
   printf("================輸出===================\n"); 
   do
    {
      Searchtime = Searchtime + 1;
      if (Temp[Middle] == Key)              //找到資料
        {
          printf("該數字是排在第 %d 個順位",Middle);      //顯示資料位置
          printf("一共搜尋 %d 次",Searchtime);            //顯示搜尋次數
         break;                               //跳出迴圈
        }
      else if(Temp[Middle] < Key) 
        Low = Middle + 1;                     //改變左半部
      else 
        High = Middle - 1;                    //改變右半部
        Middle = (int)((Low + High) / 2);     //改變中間值
    }
    while(Low <= High);
    printf("\n"); 
    system("PAUSE");
    return(0);
}
