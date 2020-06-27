#include <time.h> 
#define MAX 10
int A[10]={0};
#define SWAP(x,y) {int t; t = x; x = y; y = t;} 

int main(void) 
{ 
  printf("===============程式描述================\n");
  printf("= 程式名稱：ch8-7.c                   =\n");
  printf("= 程式目的：追蹤「謝耳排序法」的過程  =\n"); 
  printf("=======================================\n"); 
  printf("=======輸入(自動產生10個亂數值)========\n");      
  RandomNum();                //呼叫產生10個亂數值的副程式
  printf("\n=========輸出(排序後結果)==============");     
  Shellsort(A);               //呼叫謝耳排序法的副程式
  printf("\n"); 
  system("PAUSE");
  return(0);
} 
RandomNum()  //產生10個亂數值之副程式
{
  int i;
  srand((unsigned)time(NULL));   //亂數值初始化
  for (i = 0; i < MAX; i++)
    {
       A[i] = rand() % 90+10;    //產生10~100的整數亂數值
       printf("%3d",A[i]);
    }
}
 //謝耳排序法的副程式
Shellsort(int A[]) 
{ 
    int i, j, k, Gap, t; 
    Gap = MAX / 2; 
    while(Gap > 0)
     { 
        for(k = 0; k < Gap; k++) 
        { 
            for(i = k+Gap; i < MAX; i+=Gap) 
            { 
                for(j = i - Gap; j >= k; j-=Gap) 
                { 
                    if(A[j] > A[j+Gap]) 
                    { 
                        SWAP(A[j], A[j+Gap]); 
                    } 
                    else 
                        break; 
                } 
            } 
        } 
        printf("\nGap = %d：", Gap); 
        for(i = 0; i < MAX; i++) 
            printf("%d ", A[i]); 
        Gap /= 2; 
    } 
}
