#define MaxNum 10   //定義陣列的大小 
main()
{   
    int data[MaxNum],n;
    int i;
    printf("===============程式描述================\n");
    printf("= 程式名稱：ch6-3.5.c                 =\n");
    printf("= 程式目的：利用一維陣列來建立二元樹  =\n"); 
    printf("=======================================\n"); 
    printf("=================輸入==================\n");
    printf("請輸入節點個數：");
    scanf("%d",&n);
    printf("請再分別輸入這 %d 個節點的內容：\n",n);
    for (i=0;i<n;i++)
       scanf(" %d",&data[i]); 
    printf("=================輸出==================\n");
    BinaryTree_Array(data,n);  //呼叫二元樹利用一個一維陣列來儲存
    system("PAUSE");
    return(0);

}
 
//二元樹利用一個一維陣列來儲存之副程式 
BinaryTree_Array(int data[], int n)
{
    int tree[MaxNum],t;                        
    int node=0,i,temp;
    for (i=0;i<10;i++)
	   tree[i]=0;
    tree[1]=data[0];//設定樹根的值 
    temp=n;
    for (i=1;i<n;i++)
     {
      node=1;
      while(tree[node]!=0)
	  {
	   if (data[i]>tree[node])  //大於樹根的鍵值 
	     node=2*node+1;         //放到右子樹 
	   else                    //小於樹根的鍵值
	     node=2*node;          //放到左子樹 
	   if (temp<node)
		 {
		  temp=node;
		  t=temp;
		 }
	   }
	  tree[node]=data[i];
    }
    printf("二元樹儲存在一維陣列的結果：\n");
    for (i=1;i<=t;i++)
	printf("%d  ",tree[i]);
    printf("\n");
}
