#define Num 20
int data[Num]={0}; 
int BinaryTree[Num]={0}; 
main( )
{
    int n,i;
    printf("===============程式描述================\n");
    printf("= 程式名稱：ch6-4.2.c                 =\n");
    printf("= 程式目的：建立一個二元樹前序追蹤    =\n"); 
    printf("=======================================\n"); 
    printf("=================輸入==================\n"); 
    printf("請輸入節點個目：");
    scanf("%d",&n);
    printf("請再分別輸入這 %d 個節點的內容：\n",n);
    for (i=0;i<n;i++)
       scanf(" %d",&data[i]);
    CreateBinaryTree(data,n); //呼叫建立二元樹之副程式 
    printf("=================輸出==================\n");      
    printf("二元樹前序追蹤的結果：\n");
    Preorder(1);   //呼叫前序之副程式 
    printf("\n");
    system("PAUSE");
    return(0);
}

//建立二元樹之副程式 
CreateBinaryTree(int data[],int n)
 {           
    int node=1,i,j,temp;
    for (i=0;i<Num;i++)
	   BinaryTree[i]=0;  //初值設定 
    for (i=0;i<n;i++)
     {
	   BinaryTree[node]=data[i];
	   node=node+1;
       }
 }

//進行前序之副程式 
Preorder(int node)
 {
  if (BinaryTree[node]!=0)
     {
	   if (BinaryTree[node]!=0)
	   printf("%d ",BinaryTree[node]);  //列印樹根
	   Preorder(2*node);               //遞迴左子樹
	   Preorder(2*node+1);             //遞迴右子樹
      }
 }
