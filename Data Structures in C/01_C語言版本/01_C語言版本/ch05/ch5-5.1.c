#define NULL 0
typedef struct node
 {
    int data;
    struct node *link;
 }NODE;
NODE *NewNode(void);
int main(void)
 {
   printf("===============程式描述====================\n");
   printf("= 程式名稱：ch5-5.1.c                     =\n");
   printf("= 程式目的：動態配置單向鏈結串列一個節點  =\n"); 
   printf("===========================================\n");               
   int Node_data;
   NODE *list;
   list=NewNode();
   printf("=================輸入======================\n");    
   printf("請輸入單一節點內容：");
   scanf("%d",&Node_data);
   list->data=Node_data;
   list->link=NULL;
   printf("=================輸出======================\n");  
   printf("節點內容為：%d\n",list->data);        //印出串列的資料欄之內容
   system("PAUSE");             //用來表示暫停
   return(0);                        
 }
 
NODE *NewNode(void)  //建立一個新節點
{
  NODE *pt;
  pt = (NODE*)malloc(sizeof(NODE));  //動態記憶體配置
  if(pt== NULL ) {
    printf("記憶體空間不足!");
    exit(1);
  }
  return pt;
}
