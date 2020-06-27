#include <time.h>
#define Num 10
int A[10]={0};
int main()
{ //主程式
  printf("===============程式描述================\n");
  printf("= 程式名稱：ch8-3a.c                  =\n");
  printf("= 程式目的：追蹤「選擇排序法」的過程  =\n"); 
  printf("=======================================\n"); 
  printf("=================輸入==================\n");
  RandomNum();              //呼叫產生10個亂數值的副程式
  printf("\n"); 
  printf("=================輸出================================\n"); 
  SelSort(A, Num);            //呼叫選擇排序法的副程式
  printf("\n"); 
  system("PAUSE");
  return(0);
}

RandomNum()  //產生10個亂數值之副程式
{
  int i;
  srand((unsigned)time(NULL)); //亂數值初始化
  printf("產生10個亂數值：\n");
  for (i = 1; i <= Num; i++)
    {
       A[i] = rand() % 90+10;  //產生10~100的整數亂數值
       printf("%4d",A[i]);
    }
}

SelSort(int A[], int n)  //選擇排序法之副程式
  {
 int i, j, k,Temp, Min = 0;
       for (i = 1; i <= n - 1; i++)
        {
         Min = i;
         for (j = i + 1; j <= n; j++)
          if (A[j] < A[Min])
             Min = j;
          {//相鄰兩個的資料交換位置
            Temp = A[i];
            A[i] = A[Min];
            A[Min] = Temp;
          }
        printf("第 %d 次排序：", i); 
        for(k = 1; k <= n; k++) 
         {
           if(k == 1)
             if(i==1)
                printf("[%d]", A[k]); 
             else
                printf("[%d ", A[k]); 
           else if(k == i)
              printf("%d] ", A[k]); 
           else
              printf("%d  ", A[k]);    
         }         
        printf("\n");             
       }
  }
