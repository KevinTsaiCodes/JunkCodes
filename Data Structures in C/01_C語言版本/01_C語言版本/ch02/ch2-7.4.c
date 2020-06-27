main()
{
     int Sparse[4][3] = {{5, 0, 0}, {0, 0, -1}, {3, 0, 0}, {0, 9, 10}};
     int Compress[6][3];
     
     //兩個矩陣相加 
     printf("===============程式描述=======================\n");
     printf("= 程式名稱：ch2-7.4.c                        =\n");
     printf("= 程式目的：將一稀疏矩陣轉換成壓縮的表示法   =\n"); 
     printf("==============================================\n"); 
     Sparse_print(4,3,Sparse);                      //輸入稀疏矩陣
     Compress_process(4, 3, Sparse, Compress);      //壓縮處理 
     Compress_print(6, 3, Compress);                //輸出壓縮結果 
     system("PAUSE");
     return(0);
}
//輸出稀疏矩陣
Sparse_print(int m, int n, int Sparse[m][n])
{
     printf("輸入稀疏矩陣：");
     printf("\n");                     
     int i, j;
     for(i = 0; i < m; i++){
       for(j = 0; j < n; j++)
         printf("%3d ", Sparse[i][j]);
       printf("\n");
     }
}

//壓縮處理 
Compress_process(int m, int n, int Sparse[m][n], int Compress[m][n])
{
   int i,j,k;
   k = 1;                              // 設定變數初值     
   Compress[0][0] = m;                 // 陣列sparse有m列  
   Compress[0][1] = n;                 // 陣列sparse有n行 
   Compress[0][2] = 5;                 // 陣列使用5個元素  
   for ( i = 0; i < m; i++ )           // 二維陣列的走訪   
      for ( j = 0; j < n; j++ )
         if ( Sparse[i][j] != 0 )      // 判斷是否為非0元素   
         {
            Compress[k][0] = i;    // 儲存列數         
            Compress[k][1] = j;    // 儲存行數         
            //儲存元素值
            Compress[k][2] = Sparse[i][j];
            k++;                   // 下一列           
         }
}

//輸出壓縮結果
Compress_print(int m, int n, int Compress[m][n])
{
     printf("輸出壓縮結果：");
     printf("\n");             
     int i, j;
     for(i = 0; i < m; i++){
       for(j = 0; j < n; j++)
         printf("%3d ", Compress[i][j]);
       printf("\n");
     }
}
