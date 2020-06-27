#include <stdio.h>
#include <stdlib.h>
#define N 8                       //定義頂點總數 
#define Edge 10                   //定義邊的總數 
int G[N][N];                   
int Gnode[10][2] = {{1, 2}, {1, 3}, {2, 4}, {2, 5}, {3, 6}, {3, 7}, {4, 8}, {5, 8}, {6, 8}, {7, 8}};
int Visited[N];        //記錄已被拜訪過的節點 
int row, column, i;
void Create_Graph();   //宣告建立圖形
void DFS(int);         //宣告深度優先搜尋法 
int main( )
{
    printf("===============程式描述================\n");
    printf("= 程式名稱：ch7-5.1.c                 =\n");
    printf("= 程式目的：深度優先搜尋法(DFS)       =\n"); 
    printf("=======================================\n"); 
    printf("=================輸入==================\n");
 //將二維陣列的內容轉換成相鄰陣列
 int R_node;
 //相鄰陣列預設為0
 for(row = 0;row<=N - 1;row++)
  {
    for(column = 0;column<=N - 1;column++)
      G[row][column]=0;
  }
  
//設定頂點的起點
 for(row = 0;row<=Edge - 1;row++)
  {
    R_node = Gnode[row][0] - 1;     
    for(column = 0;column<=1;column++)
      G[R_node][Gnode[row][column] - 1] = 1;
  }

//設定頂點的終點
 for(row = 0;row<=Edge - 1;row++)
  {
    R_node = Gnode[row][1] - 1;
    for(column = 0;column<=1;column++)
      G[R_node][Gnode[row][column] - 1] = 1;
  }

//對角線設定為0
 for(row = 0;row<=N - 1;row++)
  {    
    for(column = 0;column<=N - 1;column++)
     {
       if(row == column)
          G[row][column]=0;
     }
  }

//設定節點都尚未被拜訪
  for(i = 0;i<=N - 1;i++)
    Visited[i] = 0;
  Create_Graph();//建立圖形 
  printf("=================輸出==================\n");        
  printf("圖形的深度優先搜尋(DFS)順序:\n");     //印出走訪過程 
  DFS(0);                                       //呼叫深先搜尋法
  printf("\n");
  system("PAUSE");	
  return 0;
}

void Create_Graph()      //建立圖形 
{
  printf("圖形的相鄰串列內容如下:\n");
  for(row = 0;row<=N - 1;row++)
  {
    for(column = 0;column<=N - 1;column++)
        printf("%3d",G[row][column]);
    printf("\n");
  }
}

void DFS(int i)    //深先搜尋法 
{
  int j;
  Visited[i] = 1;
  printf("v%d  ",i + 1);
  for(j = 0;j<=N - 1;j++)
   {
    if(G[i][j] == 1)
      if (Visited[j]!=1)  //判斷是否已經被拜訪過
           DFS(j);
   }
}
