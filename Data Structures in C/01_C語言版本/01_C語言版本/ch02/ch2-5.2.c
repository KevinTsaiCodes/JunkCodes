#define N 3
#define M 5
main(void) 
{ 
  printf("===============程式描述=======================\n");
  printf("= 程式名稱：ch2-5.2.c                        =\n");
  printf("= 程式目的：多維矩陣轉一維矩陣(以行為主)     =\n"); 
  printf("==============================================\n");             
  int Array_Data1[N][M] = {{1, 2, 3, 4, 5},
                    {6, 7, 8, 9, 10},
                    {11,12,13,14,15}};
  int Array_Data2[N*M] = {0}; 
  int row, column, i;
  printf("==================輸入========================\n"); 
  printf("二維資料之原始資料：\n");
  for(row = 0; row < N; row++)
   {
     for(column = 0; column < M; column++) 
         printf("%4d", Array_Data1[row][column]); 
     printf("\n"); 
   } 
   //處理
   for(row = 0; row < N; row++) 
   {  
      for(column = 0;column < M; column++)
       {  
         i = row + column * N;   
         Array_Data2[i] = Array_Data1[row][column]; 
       }  
    }  
  printf("==================輸出========================\n");  
  printf("以行為主：");   
  for(i = 0; i < N*M; i++)    
     printf("%d ", Array_Data2[i]);  
   printf("\n");   
   system("PAUSE");
   return(0);        
}
