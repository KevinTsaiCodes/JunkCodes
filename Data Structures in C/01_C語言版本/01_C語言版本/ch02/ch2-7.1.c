main()
{
     printf("===============程式描述=======================\n");
     printf("= 程式名稱：ch2-7.1.c                        =\n");
     printf("= 程式目的：矩陣轉置                         =\n"); 
     printf("==============================================\n");       
     int A[4][4] ={ { 1, 2, 3, 4 }, 
                    { 5, 6, 7, 8 }, 
                    { 9, 10, 11, 12 }, 
                    { 13, 14, 15, 16 } };
     int B[4][4];
     matrix_pretrans(4, 4, A);       //轉換前 
     matrix_transpose(4, 4, A, B);   //轉換副程式 
     matrix_posttrans(4, 4, B);      //轉換後 
     system("PAUSE");
     return(0);
}

//轉換前 
matrix_pretrans(int m, int n, int A[m][n])
{              
     printf("轉換前：");
     printf("\n");             
     int i, j;
     for(i = 0; i < m; i++){
       for(j = 0; j < n; j++)
         printf("%4d ", A[i][j]);
       printf("\n");
     }
}

//轉換副程式(A矩陣的m×n轉成B矩陣的n×m
matrix_transpose(int m, int n, int A[m][n], int B[n][m])
{
  int i, j;
  for(i = 0; i < m; i++)
    for(j = 0; j < n; j++)
      B[j][i] = A[i][j]; 
}

//轉換後 
matrix_posttrans(int m, int n, int B[m][n])
{
     printf("轉換後：");
     printf("\n");                   
     int i, j;
     for(i = 0; i < m; i++){
       for(j = 0; j < n; j++)
         printf("%4d ", B[i][j]);
       printf("\n");
     }
}
