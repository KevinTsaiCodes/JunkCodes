#define MaxNum 20   //定義陣列的大小 
main()
{   
    int data[MaxNum],n;
    int i;
    printf("===============程式描述================\n");
    printf("= 程式名稱：ch6-3.5a.c                =\n");
    printf("= 程式目的：利用鏈結串列來建立二元樹  =\n"); 
    printf("=======================================\n"); 
    printf("=================輸入==================\n");
    printf("請輸入節點個數：");
    scanf("%d",&n);
    printf("請再分別輸入這 %d 個節點的內容：\n",n);
    for (i=0;i<n;i++)
       scanf(" %d",&data[i]); 
    printf("=================輸出==================\n");
    BinaryTree_List(data,n);   //呼叫二元樹利用鏈結串列來儲存
    system("PAUSE");
    return(0);
}
 //二元樹利用鏈結串列來儲存之副程式 
BinaryTree_List(int data[],int n)
 {
    int tree[MaxNum],t;
    int node=0,i,temp,left_node,right_node;
    for (i=0;i<20;i++)
	   tree[i]=0;    //初值設定為0 
    tree[1]=data[0];  //設定樹根的值 
    temp=n;   //節點個數
    for (i=1;i<n;i++)
    {
	node=1;
	while(tree[node]!=0)
	     {
	      if (data[i]>tree[node])   //大於樹根的鍵值 
	        node=2*node+1;          //放到右子樹 
	     else                      //小於樹根的鍵值
	        node=2*node;           //放到左子樹 
	      if (temp<node)
		 {
		  temp=node;
		  t=temp;
		 }
	     }
	  tree[node]=data[i];
    }
    printf("二元樹利用鏈結串列來儲存方式：\n");
    printf("節點　左節點　節點之鍵值　右節點\n");
    for (i=1;i<=t;i++)
	{
	left_node=2*i;
	right_node=2*i+1;
	if (tree[left_node]==0)
	   left_node=0;
	if (tree[right_node]==0)
	   right_node=0;
	if (tree[i]!=0)
	   printf("[%d]       %d        %d       %d\n",i,left_node,tree[i],right_node);
   	}
 }
