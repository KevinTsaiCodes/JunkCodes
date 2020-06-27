#define MaxNum 20             //定義堆疊大小
char Stack[MaxNum] = {'0'};   //以陣列Stack當作堆疊 
int InfixToPostfix(char []);  //宣告中序轉後序
int Priority(char);           //宣告決定運算子優先順序 
int i = 0, Top = 0;  
int main(void) 
{    
   char itemset[20];
   printf("===============程式描述================\n");
   printf("= 程式名稱：ch3-3.4.c                 =\n");
   printf("= 程式目的：中序轉後序                =\n"); 
   printf("=======================================\n"); 
   printf("=================輸入==================\n");
   printf("請輸入中序運算式：");     
   scanf("%s", itemset);     
   printf("=================輸出==================\n");   
   InfixToPostfix(itemset);     
   return (0); 
} 
//中序轉後序之副程式 
int InfixToPostfix(char Stack[]) 
{   
   char op; 
   printf("輸出後序式結果：");    
   while(1) 
   {        
     op = Stack[i];         
     switch(op) 
     {             
         case '\0':                 
              while(Top > 0) 
              {                     
                 printf("%c", Stack[Top]);                     
                 Top--;                 
              }                 
              printf("\n");   
              system("pause");     //使程式暫停在執行畫面              
              return (0);             
         // 運算子堆疊             
         case '(':                 
              if(Top < (sizeof(Stack) / sizeof(char)))
               {                     
                 Top++;                     
                 Stack[Top] = op;                
               }                
               break;             
         case '+': case '-': case '*': case '/':
               while(Priority(Stack[Top]) >= Priority(op)) 
               {                    
                  printf("%c", Stack[Top]);                   
                  Top--;             
               }              
           // 存入堆疊                 
            if(Top < (sizeof(Stack) / sizeof(char))) 
            {                   
             Top++;                    
             Stack[Top] = op;           
            }                
            break;            
            // 遇 ) 輸出至 (     
          case ')':                 
               while(Stack[Top] != '(') 
               {                
                 printf("%c", Stack[Top]);                    
                 Top--;               
               }                 
               Top--;  // 不輸出(                 
               break;             
               // 運算元直接輸出            
          default:                
               printf("%c", op);   
               break;     
         }       
         i++;    
    }
} 
//決定運算子優先順序之副程式 
int Priority(char op) 
{     
   int p;   
   switch(op) 
   {        
      case '+': case '-':            
            p = 1;             
            break;         
      case '*': case '/':             
           p = 2;             
           break;         
      default:             
           p = 0;             
           break;   
   }    
        return p;
} 
