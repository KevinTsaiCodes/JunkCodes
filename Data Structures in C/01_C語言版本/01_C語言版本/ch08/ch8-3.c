#include <time.h>
#define Num 10
int A[10]={0};
int main()   //主程式
{ 
    printf("===============程式描述================\n");
    printf("= 程式名稱：ch8-3.c                   =\n");
    printf("= 程式目的：選擇排序法                =\n"); 
    printf("=======================================\n"); 
    printf("=======輸入(自動產生10個亂數值)========\n");    
    RandomNum();              //呼叫產生10個亂數值的副程式
    printf("\n"); 
    printf("=========輸出(排序後結果)==============\n");    
    SelSort(A, Num);          //呼叫選擇排序法的副程式
    PrintSelSort(A, Num);     //呼叫選擇排序後的結果之副程式
    printf("\n"); 
    system("PAUSE");
    return(0);
}

RandomNum()  //產生10個亂數值之副程式
{
  int i;
  srand((unsigned)time(NULL)); //亂數值初始化
  for (i = 1; i <= Num; i++)
    {
       A[i] = rand() % 90+10;  //產生10~100的整數亂數值
       printf("%4d",A[i]);
    }
}

SelSort(int A[], int n)  //選擇排序法之副程式
 {
   int i, j, Temp, Min = 0;
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
    }
 }
 
PrintSelSort(int A[], int n)  //選擇排序法之副程式 
 {
    int i;
    for (i = 1; i <= Num; i++)
    {
      printf("%4d",A[i]);
    }   
}

