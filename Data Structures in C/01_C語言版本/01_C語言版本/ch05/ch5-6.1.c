#include <stdlib.h>
void Push(int);      //宣告push副程式 
int Pop( );          //宣告pop副程式 
void PrintStack( );  //宣告列印堆疊之副程式 
typedef struct node {
  int data;
  struct node *link;
} NODE;
NODE *top=NULL;
int main(void)   //主程式
{
    printf("===============程式描述=======================\n");
    printf("= 程式名稱：ch5-6.1.c                        =\n");
    printf("= 程式目的：使用串列呈現堆疊                 =\n"); 
    printf("==============================================\n");    
  int op=0,item;
  while(1)
  {
    printf("==============================================\n");
    printf("=   1.push(加入)                             =\n");
    printf("=   2.pop(取出)                              =\n");
    printf("=   3.顯示目前Stack的內容(從由上而下Top-Down)=\n");
    printf("=   4.結束                                   =\n");    
    printf("==============================================\n");
    printf("請輸入你的操作：");
    scanf("%d",&op);
    switch(op)
    {
      case 1: printf("請輸入你要push的資料：");
              scanf("%d",&item);
              Push(item);   //呼叫push副程式 
              break;
      case 2: printf("%d 是從堆疊彈出的資料\n",Pop( ));   //呼叫pop副程式
              break;
      case 3: PrintStack( );  //呼叫列印堆疊之副程式 
              break;
      case 4: printf("\n");    //結束
              return (0);              
    }
  }
  while(top) pop( );
  system("pause");     //使程式暫停在執行畫面
  return 0;
}

//將資料加入堆疊
void Push(int data)  //push副程式
{
  NODE *pt;
  if( (pt = (NODE*) malloc(sizeof(NODE))) == NULL ) {
    printf("記憶體不足！");
    exit(1);
  }  
  pt->data = data;
  pt->link = top;
  top = pt; 
}

//取出堆疊資料
int Pop( )  //pop副程式
{
  NODE temp;
  if(top == NULL) {
    printf("堆疊是空的!");
    exit(1);
  } 
  temp = *top;
  free(top);
  top = temp.link;
  return temp.data;
}

//列印目前堆疊的內容
void PrintStack( )   //列印堆疊之副程式
{
  NODE *temp=top;   
  printf("目前堆疊的內容為: ");
  while(temp) {
    printf("%d ",temp->data);
    temp = temp->link;
  }
  printf("\n");
}
