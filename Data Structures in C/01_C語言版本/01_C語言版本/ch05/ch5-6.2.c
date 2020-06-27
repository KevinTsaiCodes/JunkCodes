#include <stdlib.h>
void Enqueue(int);       //宣告Enqueue副程式
int Dequeue(void);       //宣告Dequeue副程式
void PrintQueue(void);   //宣告列印目前佇列的內容之副程式
typedef struct node {
  int data;
  struct node *prev;
  struct node *link;
} NODE;
int flag;
NODE *front=NULL;
NODE *rear=NULL;
int main(void)
{
    printf("===============程式描述=======================\n");
    printf("= 程式名稱：ch5-6.2.c                        =\n");
    printf("= 程式目的：使用串列呈現佇列                 =\n"); 
    printf("==============================================\n");   
  int op=0,item;
  while(1)
  {
    printf("==============================================\n");
    printf("=   1.enqueue(加入)                          =\n");
    printf("=   2.dequeue(刪除)                          =\n");
    printf("=   3.顯示目前佇列的內容                     =\n");
    printf("=   4.結束                                   =\n");    
    printf("==============================================\n");
    printf("請輸入你的操作：");
    scanf("%d",&op);
    switch(op)
    {
      case 1: printf("請輸入你要加入的資料:");
              scanf("%d",&item);
              Enqueue(item);      //呼叫enqueue副程式
              break;
      case 2: item = Dequeue( );  //呼叫dequeue副程式
              printf("%d 是從佇列刪除的資料\n",item);
              break;
      case 3: PrintQueue( );  //呼叫列印佇列之副程式 
              break;
      case 4: printf("\n");       //結束
              return (0);              
    }
  }
  system("pause");     //使程式暫停在執行畫面
  return (0);
}

//資料加入佇列
void Enqueue(int data)    //enqueue副程式
{
  NODE *pt;
  if((pt = (NODE*)malloc(sizeof(NODE))) == NULL) {
    printf("記憶體不足！");
    exit(1);
  }
  if(front == NULL && rear == NULL)
    rear = pt;
  if(front != NULL) { 
    front->prev = pt;
  }    
  pt->prev = NULL;
  pt->data = data;
  pt->link = front;
  front = pt;
}

//資料從佇列刪除 
int Dequeue(void)     //dequeue副程式
{
  int pt=rear->data;
  if(flag == 1) {
    printf("佇列是空的!\n");
    exit(1);
  } else { 
    if(front == rear) {
      flag = 1;
      free(rear);
      return pt;
    }  
    rear = rear->prev;
    free(rear->link);
    rear->link = NULL;
    return pt;
  }
}
//列印目前佇列的內容
void PrintQueue(void)   //列印佇列之副程式
{
  NODE *pt=front;   
  printf("目前佇列的內容：");
  while(pt) {
    printf("%d ",pt->data);
    pt = pt->link;
  }
  printf("\n");
}
