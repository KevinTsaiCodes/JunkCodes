// app3_5, int型態的溢位處理
public class app3_5
{
   public static void main(String args[])
   {
      int i=java.lang.Integer.MAX_VALUE;     // 將i設為整數的最大值

      System.out.println("i="+i);
      System.out.println("i+1="+(i+1));         // 會發生溢位
      System.out.println("i+2="+(i+2L));
      System.out.println("i+3="+((long)i+3));
   }
}

/* app3_5 OUTPUT-----
i=2147483647
i+1=-2147483648
i+2=2147483649
i+3=2147483650
-------------------*/