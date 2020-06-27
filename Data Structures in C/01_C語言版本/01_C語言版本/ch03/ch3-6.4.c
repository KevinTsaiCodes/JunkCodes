Towers(int n, char from, char through, char to)
{
  if (n > 0){
    Towers (n - 1, from, to, through);
    printf("將盤子%d從柱子%c移到柱子%c\n", n, from, to);
    Towers (n - 1, through, from, to);
  }
}

main()
{  int n;
   printf("===============程式描述=======================\n");
   printf("= 程式名稱：ch3-6.4.c                        =\n");
   printf("= 程式目的：計算河內塔問題                   =\n"); 
   printf("==============================================\n"); 
   printf("===================輸入=======================\n"); 
   printf("請輸入盤子個數n：");  
   scanf("%d",&n);     
   printf("===================輸出=======================\n");   
   Towers (n,'A','B','C');
   printf("\n");  
   system("PAUSE");
   return(0);
}



