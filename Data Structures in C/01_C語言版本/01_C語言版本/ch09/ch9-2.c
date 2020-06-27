main()
{
  printf("===============程式描述================\n");
  printf("= 程式名稱：ch9-2.c                   =\n");
  printf("= 程式目的：循序搜尋程式1             =\n"); 
  printf("=======================================\n"); 
  printf("==============輸入(原始資料)===========\n");
  printf("90, 80, 40, 50, 65, 100, 10, 20\n");       
  int list[] = {90, 80, 40, 50, 65, 100, 10, 20};   
  printf("=================輸出==================\n"); 
  printf("%d 在陣列中的第 %d 筆.\n",100,sequential_search(list, 8, 100));
  system("PAUSE");
  return(0);
}

int sequential_search(int list[], int n, int key)
{
  int i;
  for (i = 0; i < n; i++)   //從頭到尾拜訪一次
    if (list[i] == key)		//比對陣列內的資料是否等於欲搜尋的條件
      return i+1;			//若找到符合條件的資料，就傳回其索引
    return(-1);		    	//若找不到符合條件的資料，就傳回 -1
}
