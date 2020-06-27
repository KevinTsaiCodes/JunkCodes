main()
{ //主程式
   printf("===============程式描述=======================\n");
   printf("= 程式名稱：ch3-6.2.c                        =\n");
   printf("= 程式目的：計算Fibonacci(費氏數)            =\n"); 
   printf("==============================================\n"); 
   int N = 6,Sum;
   Sum = Fib(N);                //呼叫遞迴函式
   printf("N=6時的費氏數 ");  
   printf("Sum=%2d",Sum);  
   printf("\n");  
   system("PAUSE");
   return(0);
}

int Fib(int N)  //遞迴函式名稱
  {
   if (N <= 2)
     return 1;
   else
     return Fib(N-1) + Fib(N - 2); //函式自己又可以呼叫自己
}
