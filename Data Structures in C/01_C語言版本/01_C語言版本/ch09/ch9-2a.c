main()
{ //主程式
  int list[10];
  int i,n,c,key;
  printf("===============程式描述================\n");
  printf("= 程式名稱：ch9-2a.c                  =\n");
  printf("= 程式目的：循序搜尋程式2             =\n"); 
  printf("=======================================\n"); 
  printf("========輸入(自動產生10個資料)===========\n");
  for (i = 1; i <= 10; i++)
    {
       list[i] = rand() % 90+10;  //產生10~100的整數亂數值
       printf("%4d",list[i]);
    }
  printf("\n");
  printf("=================輸出====================\n"); 
  printf("%d 在陣列中的第 %d 筆.\n",28,sequential_search(list, 10, 28));
  system("PAUSE");
  return(0);

}    

int sequential_search(int list[], int n, int key)
{
  int i;
  for (i = 1; i <= n; i++)
    if (list[i] == key) 	/* 比對陣列內的資料是否等於欲搜尋的條件 */
      return i;			    /* 若找到符合條件的資料，就傳回其索引 */
    return(-1);		        /* 若找不到符合條件的資料，就傳回 -1 */
}
