main()
{
  printf("===============程式描述=======================\n");
  printf("= 程式名稱：ch5-3.c                          =\n");
  printf("= 程式目的：動態記憶體配置及釋放記憶體       =\n"); 
  printf("==============================================\n");     
  int *pt;         //宣告一個整數的指標p
  // 配置記憶體給pt，使pt成為一個整數陣列
  printf("動態記憶體配置:\n");
  pt = (int *) malloc(sizeof(int));
  if(pt==0) //配置記憶體失敗
    return;
  pt[0] = 20; 
  pt[1] = 30; 
  pt[2] = 40;
  int i;
  for(i=0;i<3;i++) 
    printf("pt[%d]=%2d\n",i,pt[i]);
  printf("free來釋放記憶體：\n");   
 // system("pause");
  for(i=0;i<3;i++) 
     free(pt++); // 釋放記憶體
  for(i=0;i<3;i++) 
    printf("pt[%d]=%2d\n",i,pt[i]);
  system("pause");
}
