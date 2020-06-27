#include <stdio.h>
typedef struct node {
  int data;
  struct node *link;
} NODE;
NODE *NewNode(void);       //宣告建立新節點 
void PrintLists(NODE*);    //宣告列印兩個或兩個以上串列的內容 
int main(void)
{   
    printf("===============程式描述====================\n");
    printf("= 程式名稱：ch5-5.2.c                     =\n");
    printf("= 程式目的：兩個或兩個以上節點的鏈結串列  =\n"); 
    printf("===========================================\n");      
    int i;
    char choice;
    NODE *head,*temp;      //宣告head,temp為指向結構NODE的指標 
    head = NewNode( );     //分配新節點空間 
    temp = head;           //兩個指標指向同一個節點 
    printf("=================輸入======================\n");     
    printf("請輸入串列第1個資料欄位值：");
    scanf("%d",&head->data);
    for(i=1;i<=4;i++) 
    {
      printf("請輸入串列第%d個資料欄位值：",i+1);
      temp->link = NewNode( );  //呼叫建立新節點 
      scanf("%d",&temp->link->data);
      temp = temp->link;
    }
    temp->link = NULL;
    printf("=================輸出======================\n");      
    PrintLists(head);   //呼叫列印兩個或兩個以上串列的內容
    while(head != NULL) {
      NODE *temp;
      temp = head;
      head = head->link;
      free(temp);
    }
  system("pause");     //使程式暫停在執行畫面
  return 0;
}
NODE *NewNode(void)  //建立新節點之副程式 
{
  NODE *pt;
  pt = (NODE*)malloc(sizeof(NODE));
  if(pt== NULL ) {
    printf("記憶體空間不足!");
    exit(1);
  }
  return pt;
}

void PrintLists(NODE* head)  //列印兩個或兩個以上串列的內容之副程式 
{
  NODE *temp=head;
  printf("鏈結串列為：");        //印出串列的資料欄之內容
  while(temp != NULL)
  {
    printf("%d ",temp->data);
    temp = temp->link;
  }
  printf("\n");
}
