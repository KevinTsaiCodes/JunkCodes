#include <time.h> 
#define MAX1 10 
#define MAX2 10 
int A[10]={0};
int B[10]={0};
int C[20]={0};
#define SWAP(x,y) {int t; t = x; x = y; y = t;} 

int main(void) 
{ 
  printf("===============程式描述================\n");
  printf("= 程式名稱：ch8-8.c                   =\n");
  printf("= 程式目的：追蹤「合併排序法」的過程  =\n"); 
  printf("=======================================\n"); 
  printf("=======輸入(自動產生10個亂數值)========\n");     
  RandomNum();                //呼叫產生10個亂數值的副程式
  printf("\n"); 
  Quicksort(A, 0, MAX1-1);    //第一組資料排序 
  Quicksort(B, 0, MAX2-1);    //第二組資料排序
  PrintQuicksort(A,B);        //列印出第一與二組資料排序結果 
  printf("\n"); 
  Mergesort(A,MAX1,B,MAX2,C); //進行合併排序 
  printf("\n"); 
  PrintMergesort(C);          //顯示合併排序之結果 
  printf("\n"); 
  system("PAUSE");
  return(0);
} 
RandomNum()  //產生10個亂數值之副程式
{
  int i;
  srand((unsigned)time(NULL));   //亂數值初始化
  printf("產生10個亂數值：\n"); 
  printf("第一組資料：");
  for (i = 0; i < MAX1; i++)
    {
       A[i] = rand() % 40+10;    //產生10~50的整數亂數值
       printf("%3d",A[i]);
    }
  printf("\n"); 
   printf("第二組資料：");
  for (i = 0; i < MAX2; i++)
    {
       B[i] = rand() % 50+51;    //產生51~100的整數亂數值
       printf("%3d",B[i]);
    }   
}

Quicksort(int A[], int left, int right)
 { 
    int q; 
    if(left < right) 
    { 
        q = partition(A, left, right); 
        Quicksort(A, left, q-1); 
        Quicksort(A, q+1, right); 
    } 
} 
//分割之副程式 
partition(int A[], int left, int right) { 
    int i, j, s; 
    s = A[right]; 
    i = left - 1; 
    for(j = left; j < right; j++) { 
        if(A[j] <= s) { 
            i++; 
            SWAP(A[i], A[j]); 
        } 
    } 
    SWAP(A[i+1], A[right]); 
    return i+1; 
} 

//快速排序法之副程式 
PrintQuicksort(int A[], int n)  
 {
    int i;
    printf("\n=========輸出(排序後結果)==============");  
    printf("\n第一組資料："); 
    for(i = 0; i < MAX1; i++) 
        printf("%d ", A[i]); 
    printf("\n第二組資料："); 
    for(i = 0; i < MAX2; i++) 
        printf("%d ", B[i]); 
}
//進行合併排序
Mergesort(int A[], int M, int B[],int N, int C[])
 { 
    int i = 0, j = 0, k = 0; 
    while(i < M && j < N) 
    { 
        if(A[i] <= B[j]) 
            C[k++] = A[i++]; 
        else 
            C[k++] = B[j++]; 
    } 
    while(i < M) 
        C[k++] = A[i++]; 
    while(j < N) 
        C[k++] = B[j++]; 
}
//顯示合併排序之結果
PrintMergesort(int C[])  //合併後之副程式 
 {
    int i;
    printf("合併後：\n"); 
    for(i = 0; i < MAX1+MAX2; i++) 
        printf("%d ", C[i]); 

}
