#include <time.h>
#define NUM 10
int BinaryTree[NUM]={0};
main()
{
    printf("===============程式描述================\n");
    printf("= 程式名稱：ch6-3.7.c                 =\n");
    printf("= 程式目的：建立二元樹                =\n"); 
    printf("=======================================\n");       
    int i,BinaryTree[NUM+1];
    srand((unsigned)time(NULL));
    printf("=================輸入==================\n"); 
    printf("亂數自動產生十個數值：\n");     
    for(i=1;i<11;i++) 
    {
       BinaryTree[i] = rand( ) % 100;
       printf("%d  ",BinaryTree[i]);
    }
    printf("\n=================輸出==================\n");   
    printf("樹根的鍵值為 %d\n",BinaryTree[1]);
    for(i=1;i<=NUM;i++)
    {
     if(2*i <= NUM) 
       printf("節點 %d 的左子樹為：%d\n",BinaryTree[i],BinaryTree[2*i]);
     if(2*i < NUM)
       printf("節點 %d 的右子樹為：%d\n",BinaryTree[i],BinaryTree[2*i+1]);
    }
    system("pause");     //使程式暫停在執行畫面
    return (0);
}

