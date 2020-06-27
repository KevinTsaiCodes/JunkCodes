#define NUM 10          //定義佇列大小
void Enqueue(int);      //宣告Enqueue副程式
int Dequeue(void);      //宣告Dequeue副程式
void PrintQueue(void);  //宣告列印目前佇列的內容之副程式
typedef struct queue {
  int Q[NUM];
  int Rear;
  int Front;
} Queue;
Queue qu;
int main(void)   //主程式
{
  int op=0,item;
  qu.Rear = -1;
  qu.Front = -1;
    printf("===============程式描述=======================\n");
    printf("= 程式名稱：ch4-2.c                          =\n");
    printf("= 程式目的：使用佇列進行Enqueue以及Dequeue   =\n"); 
    printf("==============================================\n");   
  while(1)
  {
    printf("==============================================\n");
    printf("=   1.Enqueue(加入)                          =\n");
    printf("=   2.Dequeue(刪除)                          =\n");
    printf("=   3.顯示目前佇列的內容                     =\n");
    printf("=   4.結束                                   =\n");    
    printf("==============================================\n");
    printf("請輸入你的操作：");
    scanf("%d",&op);
    switch(op)
    {
      case 1: printf("請輸入你要加入的資料:");
              scanf("%d",&item);
              Enqueue(item);      //呼叫Enqueue副程式 
              break;
      case 2: item = Dequeue( );  //呼叫Dequeue副程式
              if(item != -1)
                printf("%d 是從佇列刪除的資料\n",item);
              break;
      case 3: PrintQueue( );      //呼叫PrintQueue副程式
              break;
      case 4: printf("\n");       //結束
              return (0);  
    }
  }
  system("pause");     //使程式暫停在執行畫面
  return (0);
}

//資料加入佇列
void Enqueue(int item)    //Enqueue副程式 
{
  if(qu.Rear == NUM -1) {
    printf("佇列是滿的!\n");
    return;
  } else
    qu.Q[++qu.Rear] = item;
}

//資料從佇列刪除 
int Dequeue(void)        //Dequeue副程式
{
  if(qu.Front == qu.Rear) {
    printf("佇列是空的!\n");
    return -1;
  } else
    return qu.Q[++qu.Front];
}

//列印目前佇列的內容
void PrintQueue(void)     //列印目前佇列的內容之副程式
{
  int i;
  printf("目前佇列的內容：");
  for(i=qu.Rear;i>=qu.Front+1;i--)
    printf("%d ",qu.Q[i]);
  printf("\n");  
}
