main()
{ //祘Α
   printf("===============祘Α磞瓃=======================\n");
   printf("= 祘Α嘿ch3-6.3.c                        =\n");
   printf("= 祘Αヘノ患癹―程そ计ぇ祘Α   =\n"); 
   printf("==============================================\n"); 
   int a=18,b=15;
   printf("a=18,b=15ぇ"); 
   printf("程そ计=%2d",GCD(a, b));   //㊣患癹ㄧΑ
   printf("\n");
   system("PAUSE");
   return(0);
}

int GCD(int a,int b)  //患癹ㄧΑ
  {
    int c;
    c = a % b;
    if (c == 0)
        return b;
    else
        return GCD(b, c);  //ㄧΑ㊣
  }

