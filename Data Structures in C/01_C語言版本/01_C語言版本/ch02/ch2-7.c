main()
{
  printf("===============程式描述=======================\n");
  printf("= 程式名稱：ch2-7.c                          =\n");
  printf("= 程式目的：印出矩陣中的所有元素             =\n"); 
  printf("==============================================\n");       
  int A[2][3] = {{1, 2, 3}, {4, 5, 6}};
  matrix_traverse(2, 3, A);
  system("PAUSE");
  return(0);
}

//假設A為2x3陣列，印出二維陣列內所有元素的值
matrix_traverse(int m, int n, int A[m][n])
{
     int i, j;
     for(i = 0; i < m; i++)
     {
       for(j = 0; j < n; j++)
         printf("%d ", A[i][j]);
       printf("\n");
     }
}
