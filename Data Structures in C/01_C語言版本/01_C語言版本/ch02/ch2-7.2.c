main()
{
     printf("===============程式描述=======================\n");
     printf("= 程式名稱：ch2-7.2.c                        =\n");
     printf("= 程式目的：2個矩陣相加                      =\n"); 
     printf("==============================================\n");       
     int A[2][3] = {{1, 2, 3}, {4, 5, 6}};
     int B[2][3] = {{11, 12, 13}, {14, 15, 16}};
     int C[2][3];
     //兩個矩陣相加 
     printf("==================輸入========================\n");       
     matrix_Aprint(2,3,A);         //輸出A矩陣
     matrix_Bprint(2,3,B);         //輸出B矩陣  
     matrix_Add(2, 3, A, B, C);     //C=A+B
     printf("==================輸出========================\n");    
     matrix_Cprint(2, 3, C);        //輸出c矩陣 
     system("PAUSE");
     return(0);
}

//輸出A矩陣
matrix_Aprint(int m, int n, int A[m][n])
{
     printf("輸出A矩陣：");
     printf("\n");                     
     int i, j;
     for(i = 0; i < m; i++){
       for(j = 0; j < n; j++)
         printf("%3d ", A[i][j]);
       printf("\n");
     }  
       printf("\n");
}

//輸出B矩陣
matrix_Bprint(int m, int n, int B[m][n])
{
     printf("輸出B矩陣：");
     printf("\n");                     
     int i, j;
     for(i = 0; i < m; i++){
       for(j = 0; j < n; j++)
         printf("%3d ", B[i][j]);
       printf("\n");
     }
       printf("\n");     
}

/*假設A,B,C均為mxn陣列，這個函數要求出C=A+B*/
matrix_Add(int m, int n, int A[m][n], int B[m][n], int C[m][n])
{
  int i, j;
  for(i = 0; i < m; i++)
    for(j = 0; j < n; j++)
      C[i][j] = A[i][j] + B[i][j];  
}

//輸出相加結果
matrix_Cprint(int m, int n, int C[m][n])
{
     printf("輸出A+B=C的結果：");
     printf("\n");             
     int i, j;
     for(i = 0; i < m; i++){
       for(j = 0; j < n; j++)
         printf("%3d ", C[i][j]);
       printf("\n");
     }
}
