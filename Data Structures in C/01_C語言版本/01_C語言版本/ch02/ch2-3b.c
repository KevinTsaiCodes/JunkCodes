main()
{  
 printf("===============祘Α磞瓃=======================\n");
 printf("= 祘Α嘿ch2-3b.c                         =\n");
 printf("= 祘Αヘ璸衡–ヘキАだ计     =\n"); 
 printf("==============================================\n");    
 //の砞﹚    
 int i,j;
 int Sum[5];  
 int Score[4][5] = {{ 65, 85, 78, 75, 69 }, { 66, 55, 52, 92, 47 }, { 75, 99, 63, 73, 86 }, { 77, 88, 99, 91, 100 } };
 //矪瞶
 for (j = 0; j <=4; j++)                //北︽计
  Sum[j] = 0;                           //盢Sum皚耴0 
 for (j = 0; j <=4; j++)                //北︽计    
   for (i = 0; i <=3; i++)              //北计
      Sum[j] = Sum[j] + Score[i][j];    //璸衡–ヘ羆だ计
 //块
 for (j = 0; j <= 4; j++)      
 {  //璸衡–ヘキАだ计
  printf("材%dヘキАだ计=%d",j+1,Sum[j]/4);   
  printf("\n");  
 }
 system("PAUSE");
 return(0);
}

