// app5_4, 區域變數
public class app5_4
{
   public static void main(String args[])
   {
      int sum=0;

      for(int i=1;i<=5;i++)   // 在迴圈內宣告變數i
      {
         sum=sum+i;
         System.out.println("i="+i+", sum="+sum);
      }
   }
}

/* app5_4 OUTPUT------
i=1, sum=1
i=2, sum=3
i=3, sum=6
i=4, sum=10
i=5, sum=15
--------------------*/