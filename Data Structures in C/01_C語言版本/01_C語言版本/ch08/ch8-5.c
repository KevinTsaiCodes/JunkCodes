#include <time.h>
#define MAX 10
int A[10]={0};
int main()   //主程式
{ 
  printf("===============程式描述================\n");
  printf("= 程式名稱：ch8-5.c                   =\n");
  printf("= 程式目的：追蹤「快速排序法」的過程  =\n"); 
  printf("=======================================\n"); 
  printf("=================輸入==================\n");
  RandomNum();              //呼叫產生10個亂數值的副程式
  printf("\n"); 
  printf("=================輸出==================\n");  
  QuickSort(A, 0, MAX-1, MAX);       //呼叫快速排序法的副程式
  printf("\n"); 
  system("PAUSE");
  return(0);
}

RandomNum()  //產生10個亂數值之副程式
{
  int i;
  srand((unsigned)time(NULL)); //亂數值初始化
  printf("產生10個亂數值：\n");   
  for (i = 0; i <= MAX-1; i++)
    {
       A[i] = rand() % 90+10;  //產生10~100的整數亂數值
       printf("%4d",A[i]);
    }
}

QuickSort(int A[], int left, int right,int index)
{  int lower,upper,t,k,key,temp,done;
   lower=left; 
   upper=right+1;                      
   t=1;
   key=A[left];                      
   if  ( lower < upper )
   { do
      {
	   do                              
        { lower ++;
	    } while (A[lower] <= key && lower <= right);
	   do                                
        { upper--;
	    } while (A[upper] >= key && upper > left);

	   if ( lower < upper )		     
	   {
	       temp=A[lower];
	       A[lower]=A[upper];
	       A[upper]=temp;
	    }
     }while ( lower < upper );
   if ( lower > upper)
       printf("\n"); 
       {
        temp=A[left];          
	    A[left]=A[upper];
	    A[upper]=temp;
        for (k=0;k<index;k++)
          printf("%d ",A[k]);
       }
        QuickSort(A, left, upper-1,index);   // 對左邊進行遞迴 
        QuickSort(A, upper+1, right,index);  // 對右邊進行遞迴 
    } 
}
