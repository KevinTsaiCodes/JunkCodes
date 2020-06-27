main()
{ //主程式
   printf("===============程式描述=======================\n");
   printf("= 程式名稱：ch3-6.1.c                        =\n");
   printf("= 程式目的：計算遞迴函數呼叫10!              =\n"); 
   printf("==============================================\n"); 
   int Sum, Max = 10;
   Sum = fact(Max);                //呼叫遞迴函式
   printf("10!=1*2*...*10=%3d",Sum);  
   printf("\n"); 
   system("PAUSE");
   return(0);
}

int fact(int N)  //遞迴函式名稱
 {
   if (N ==1)
     return 1;
   else
     return N * fact(N - 1); //函式自己又可以呼叫自己
 }
