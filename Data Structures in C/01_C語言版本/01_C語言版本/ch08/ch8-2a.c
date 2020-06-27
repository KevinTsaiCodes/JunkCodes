#include <time.h>
#define Num 10
int A[10]={0};
int main()
{ //主程式
  printf("===============程式描述================\n");
  printf("= 程式名稱：ch8-2a.c                  =\n");
  printf("= 程式目的：追蹤「氣泡排序法」的過程  =\n"); 
  printf("=======================================\n"); 
  printf("=================輸入==================\n");
  RandomNum();              //呼叫產生10個亂數值的副程式
  printf("\n"); 
  printf("=================輸出================================\n");  
  BubSort(A, Num);            //呼叫氣泡排序法的副程式
  printf("\n"); 
  system("PAUSE");
  return(0);
}

RandomNum()  //產生10個亂數值之副程式
{
  int i;
  srand((unsigned)time(NULL)); //亂數值初始化
  printf("產生10個亂數值：\n");
  for (i = 0; i <= Num-1; i++)
    {
       A[i] = rand() % 90+10;  //產生10~100的整數亂數值
       printf("%4d",A[i]);
    }
}

BubSort(int A[], int n)  //氣泡排序法之副程式
  {
    int i, j , k,t, Temp;
    t = 0;
    for (i =n-1; i>0; i--)
       {
       for (j =0; j <=i-1; j++)
          if (A[j] > A[j+1])
             {
               Temp = A[j];
               A[j] = A[j+1];
               A[j+1] = Temp;
             }
        t = t + 1;     
        printf("第 %d 次排序：", t); 
        for(k = 0; k <= n-1; k++) 
         {
           if(k == (n - t - 1))
              printf("%d [", A[k]); 
           else if(k == n - 1)
              printf("%d]", A[k]); 
           else
              printf("%d  ", A[k]);    
         }
        printf("\n");             
       }
  }
