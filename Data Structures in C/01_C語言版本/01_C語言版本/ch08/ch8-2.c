#include <time.h>
#define Num 10
int A[10]={0};
main()
{ //主程式
    printf("===============程式描述================\n");
    printf("= 程式名稱：ch8-2.c                   =\n");
    printf("= 程式目的：氣泡排序法                =\n"); 
    printf("=======================================\n"); 
    printf("=======輸入(自動產生10個亂數值)========\n");
    RandomNum();              //呼叫產生10個亂數值的副程式
    printf("\n"); 
    printf("=========輸出(排序後結果)==============\n");
    BubSort(A, Num);          //呼叫氣泡排序法的副程式
    PrintBubSort(A, Num);     //呼叫排序後的結果之副程式
    printf("\n"); 
    system("PAUSE");
    return(0);
}

RandomNum()  //氣泡排序法之副程式
{
  int i;
  srand((unsigned)time(NULL));
  for (i = 1; i <= Num; i++)
    {
       A[i] = rand() % 90+10;  //產生10~100的整數亂數值
       printf("%3d",A[i]);
    }
}


BubSort(int A[], int n)  //氣泡排序法之副程式
  {
    int i, j , k,t=1, Temp;
    for (i=n-1; i>0; i--)
       {
       for (j =0; j <=i; j++)
          if (A[j] > A[j+1])
             {  //兩數交換位置
               Temp = A[j];
               A[j] = A[j+1];
               A[j+1] = Temp;
             }          
       }
  }
  
PrintBubSort(int A[], int n)  //氣泡排序法之副程式 
 {
    int i;
    for (i = 1; i <= Num; i++)
    {
      printf("%3d",A[i]);
    }   
}
