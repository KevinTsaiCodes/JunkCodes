#include <time.h>
#define Num 10
int data[10] = {79, 8, 6, 93, 59, 84, 55, 9, 71, 33}; 
int temp[10][10] = {0}; 
int order[10] = {0}; 
int i, j, k, n, lsd; 
int main()
{ //主程式
  printf("===============程式描述================\n");
  printf("= 程式名稱：ch8-9.c                   =\n");
  printf("= 程式目的：追蹤「合併排序法」的過程  =\n"); 
  printf("=======================================\n"); 
  printf("===========輸入(10個原始資料)==========\n"); 
    k = 0; 
    n = 1; 
  ShowData();                  //呼叫顯示原始資料
  printf("\n"); 
  printf("\n============輸出(排序後結果)=============="); 
  Radix(data,n);                 //呼叫基數排序法的副程式
  printf("\n"); 
  system("PAUSE");
  return(0);
}

ShowData ()  //顯示原始資料
{
    int i;
    printf("排序前: "); 
    for(i = 0; i < 10; i++) 
        printf("%d ", data[i]); 
}
//基數排序法的副程式
Radix(int data[],int n)  
{   int t=1;
    while(n <= 10)
     { 
        for(i = 0; i < 10; i++) 
        { 
            lsd = ((data[i] / n) % 10); 
            temp[lsd][order[lsd]] = data[i]; 
            order[lsd]++; 
        } 
        if (t==1) 
           printf("\n「個位數」為主："); 
        else 
           printf("\n「十位數」為主："); 
        for(i = 0; i < 10; i++)
         { 
            if(order[i] != 0) 
                for(j = 0; j < order[i]; j++) 
                { 
                    data[k] = temp[i][j]; 
                    printf("%d ", data[k]); 
                    k++; 
                } 
            order[i] = 0; 
        } 
        n *= 10; 
        k = 0; 
        t+=1;
    } 
  }
