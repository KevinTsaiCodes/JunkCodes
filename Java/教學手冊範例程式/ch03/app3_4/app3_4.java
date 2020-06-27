// app3_4, 整數資料型態的溢位
public class app3_4
{
   public static void main(String args[])
   {
      int i=java.lang.Integer.MAX_VALUE;     // 將i設為整數的最大值
      int sum;

      System.out.println("i="+i);            // 印出i的值

      sum=i+1;
      System.out.println("i+1="+sum);        // 印出i+1的值

      sum=i+2;
      System.out.println("i+2="+sum);        // 印出i+2的值
   }
}

/* app3_4 OUTPUT------------
i=2147483647
i+1=-2147483648
i+2=-2147483647
---------------------------*/