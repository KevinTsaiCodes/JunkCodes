main( )
{
  printf("===============程式描述=======================\n");
  printf("= 程式名稱：ch1-4.c                          =\n");
  printf("= 程式目的：計算平均成績，並判斷是否及格     =\n"); 
  printf("==============================================\n"); 
  int C_Score, E_Score, Average;
  C_Score=60;
  E_Score=70;
  Average = (C_Score + E_Score) / 2;
  printf("平均成績：%d\n",Average);
  printf("結果：");
  if (Average >= 60)
     printf("及格\n");
  else
     printf ("不及格\n");
  system("PAUSE");	
  return 0;
}
