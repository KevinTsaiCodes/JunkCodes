#define MaxNum 5           //定義堆疊大小
int Stack[MaxNum];         //以陣列Stack當作堆疊
int Top = -1;  //Top紀錄目前堆疊頂端的索引值，初始值設為-1表示堆疊為空
void Push(int);            //宣告Push(加入)副程式 
int Pop(void);             //宣告Pop(取出)副程式 
void PrintStack(void);     //宣告列印目前堆疊的內容之副程式 
int main(void)             //主程式 
{
   int op=0,item;   
    printf("===============程式描述=======================\n");
    printf("= 程式名稱：ch3-2.c                          =\n");
    printf("= 程式目的：使用堆疊進行push以及pop          =\n"); 
    printf("==============================================\n");    
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
      case 1: printf("請輸入你要push(加入)的資料：");
              scanf("%d",&item);
              Push(item);     //呼叫Push副程式 
              break;
      case 2: printf("%d 是從堆疊彈pop(取出)的資料\n",Pop( )); //呼叫Pop副程式 
              break;
      case 3: PrintStack( );  //顯示目前Stack的內容 
              break;
      case 4: printf("\n");    //結束
              return (0);            
    }
  }
  system("pause");     //使程式暫停在執行畫面
  return (0);
}

//將資料加入堆疊
void Push(int item)    //Push(加入)副程式 
{
  if(Top == MaxNum -1)
    printf("堆疊是滿的!");
  else
    Stack[++Top] = item;
}

//取出堆疊資料
int Pop(void)         //Pop(取出)副程式
{
  if(Top == -1) 
    printf("堆疊是空的!");
  else
    return Stack[Top--];
}

//列印目前堆疊的內容
void PrintStack(void)    //列印堆疊之副程式
{
  int i;
  if(Top == -1) 
    printf("堆疊是空的!\n");
  else 
  {
    printf("目前堆疊的內容為: ");
    for(i=Top;i>=0;i--)
      printf("%d ",Stack[i]);
    printf("\n");
  }
}
