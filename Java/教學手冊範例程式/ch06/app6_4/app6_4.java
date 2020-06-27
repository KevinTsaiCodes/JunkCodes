// app6_4, 二維陣列的輸入輸出
public class app6_4
{
   public static void main(String args[])
   {
      int i,j,sum=0;
      int sale[][]={{32,35,26,30},{34,30,33,31}}; // 宣告陣列並設定初值

      for(i=0;i<sale.length;i++) // 輸出銷售量並計算總銷售量
      {
         System.out.print("業務員"+(i+1)+"的業績分別為 ");
          for(j=0;j<sale[i].length;j++)
          {
             System.out.print(sale[i][j]+" ");
             sum+=sale[i][j];
          }
          System.out.println();        // 列印換行
      }
      System.out.println("\n總銷售量為"+sum+"部車");
   }
}

/* app6_4 OUTPUT----------------
業務員1的業績分別為 32 35 26 30
業務員2的業績分別為 34 30 33 31

總銷售量為251部車
------------------------------*/