#include <time.h> 
#define MAX 10
int A[10]={0};
int main(void) { 
  printf("===============程式描述================\n");
  printf("= 程式名稱：ch8-6.c                   =\n");
  printf("= 程式目的：追蹤「堆積排序法」的過程  =\n"); 
  printf("=======================================\n"); 
  printf("=======輸入(自動產生10個亂數值)========\n");  
  int number[MAX+1] = {-1}; 
  int i, num;  
  RandomNum();                //呼叫產生10個亂數值的副程式
  printf("\n"); 
  printf("\n建立堆積樹：\n"); 
  Heapsort(A,MAX);                //呼叫堆積排序法的副程式
  printf("\n"); 
  system("PAUSE");
  return(0);
} 

RandomNum()  //產生10個亂數值之副程式
{
  int i;
  srand((unsigned)time(NULL)); //亂數值初始化    
  for (i = 1; i <= MAX; i++)
    {
      A[i] = rand() % 90+10;  //產生10~100的整數亂數值
      printf("%4d",A[i]);
    }
}

//呼叫堆積排序法的副程式
Heapsort(int A[], int index)
{  int i, j,t, temp;
    t=1;
    //將完整二元樹轉成堆積樹 
    for ( i= (index/2); i >= 1; i--)
       Createheap(A, i , index);
    for(i = 1; i <= 10; i++) 
       printf("%4d",A[i]);
    //累堆排序
    printf("\n=========輸出(排序後結果)=============="); 
    for  ( i = index-1; i >= 1; i--)
    {
	 temp=A[i+1];          
     A[i+1]=A[1];
     A[1]=temp;
	 Createheap(A,1,i);    
     printf("\n第 %d 次排序：", t++);
     for ( j=1; j <= index; j++)
	       printf("%d ", A[j] );
   }
}

//建立堆積樹的副程式
Createheap(int A[],int root, int index)
{    int i, j , temp;
     int finish;              
     j=2*root;                
     temp=A[root];            
     finish=0;                
     while ( j<= index  &&  finish==0 )
     {
     //找最大值
	 if ( j < index )
	       if ( A[j] < A[j+1] )
              j++;
	 if ( temp >= A[j] )
	     finish=1;         
        else
        {
	     A[j/2]=A[j];      
         j=2*j;
        }
     }
     A[j/2]=temp;        
} 
