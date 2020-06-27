// app7_7, 設計傳回二維陣列的method
public class app7_7
{
   public static void main(String args[])
   {
     int A[][]={{18,32,65,27,30},{17,56,12,66}};  // 宣告二維陣列A
     int B[][];

     B=add10(A);   // 呼叫add10()，並把傳回的值設給陣列B
     for(int i=0;i<B.length;i++) // 印出陣列的內容
     {
        for(int j=0;j<B[i].length;j++)
           System.out.print(B[i][j]+" ");
        System.out.println();
     }
  }

  public static int[][] add10(int arr[][])
  {
     for(int i=0;i<arr.length;i++)
        for(int j=0;j<arr[i].length;j++)
           arr[i][j]+=10;        // 將陣列元素加10
     return arr;              // 傳回二維陣列
  }
}

/* app7_7 OUTPUT---
28 42 75 37 40
27 66 22 76
-----------------*/