#include <stdio.h>
#include <stdlib.h> 
#define M 5 
#define N 5 
 int main(void)
  { 
    printf("===============程式描述=======================\n");
    printf("= 程式名稱：ch2-8.1.c                        =\n");
    printf("= 程式目的：上三角矩陣轉一維矩陣(以列為主)   =\n"); 
    printf("==============================================\n");            
    int Array_Data1[M][N] = 
    { 
      {1, 2, 3,  4,   5},
      {0, 6, 7,  8,   9},
      {0, 0, 10, 11, 12},
      {0, 0, 0,  13, 14},
      {0, 0, 0,  0,  15}};
    int Array_Data2[N*(1+N)/2] = {0};  //元素個數(非0元素)
    int i, j, Location = 0; 
    printf("==================輸入========================\n");      
    printf("上三角矩陣的原始資料：\n");
    for(i = 0; i < M; i++) 
    { 
      for(j = 0; j < N; j++)
       {       
         printf("%4d", Array_Data1[i][j]); 
       }       
        printf("\n"); 
     }   
     for(i = 0; i < M; i++) 
     {    
       for(j = 0; j < N; j++)
        {      
          if(Array_Data1[i][j] != 0)   //非0元素
           Array_Data2[Location++] = Array_Data1[i][j]; 
        } 
      }   
   printf("==================輸出========================\n");    
   printf("以列為主：");   
   for(i = 0; i < N*(1+N)/2; i++) 
     printf("%d ", Array_Data2[i]);  
   printf("\n");   
   system("PAUSE");
   return(0);
}
