main()
{
     printf("===============程式描述=======================\n");
     printf("= 程式名稱：ch2-7.3.c                        =\n");
     printf("= 程式目的：2個矩陣相乘                      =\n"); 
     printf("==============================================\n"); 
     int A[2][3] = {{1, 2, 3}, {4, 5, 6}};
     int B[3][3] = {{1, 0, 1}, {1, 1, 0}, {0, 1, 1}};
     int C[2][3];
     //兩個矩陣相乘 
     printf("==================輸入========================\n"); 
     matrix_Aprint(2,3,A);         //輸出A矩陣
     matrix_Bprint(3,3,B);         //輸出B矩陣  
     matrix_Mul(2, 3, 3, A, B, C);   //C=A*B
     printf("==================輸出========================\n"); 
     matrix_Cprint(2, 3, C);      //輸出c矩陣 
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

/*假設A,B,C均為mxn陣列，這個函數要求出C=A*B*/
matrix_Mul(int m, int n, int p, int A[m][n], int B[n][p], int C[m][p])
{
  int i, j, k;
  for(i = 0; i < m; i++)
    for(j = 0; j < n; j++)
      {
        C[i][j]=0;
        for(k=0;k<n;k++)
        C[i][j] = C[i][j] + A[i][k] * B[k][j]; 
      }
}

//輸出相乘結果
matrix_Cprint(int m, int n, int C[m][n])
{
     printf("輸出A*B=C的結果：");
     printf("\n");             
     int i, j;
     for(i = 0; i < m; i++){
       for(j = 0; j < n; j++)
         printf("%3d ", C[i][j]);
       printf("\n");
     }
}
