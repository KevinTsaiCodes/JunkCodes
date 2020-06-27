#include <time.h>
#define Num 10
int A[10]={0};
int main()   //主程式
{ 
    printf("===============程式描述================\n");
    printf("= 程式名稱：ch8-4.c                   =\n");
    printf("= 程式目的：插入排序法                =\n"); 
    printf("=======================================\n"); 
    printf("=======輸入(自動產生10個亂數值)========\n");      
    RandomNum();              //呼叫產生10個亂數值的副程式
    printf("\n"); 
    printf("=========輸出(排序後結果)==============\n");  
    InSort(A, Num);          //呼叫插入排序法的副程式
    PrintInSort(A, Num);     //呼叫插入排序後的結果之副程式
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

InSort(int A[], int n)  //插入排序法之副程式
 {
   int i, j, Temp;
   for (i = 1; i <= n; i++)
    {
     Temp=A[i];
     j=i-1;
     while (Temp<A[j])
      {
        A[j+1]=A[j];
        j--;
        if(j==-1)
          break;
      }
      A[j+1]=Temp;
    }
 }
 
PrintInSort(int A[], int n)  //插入排序法之副程式 
 {
    int i;
    for (i = 1; i <= Num; i++)
    {
      printf("%4d",A[i]);
    }   
}

