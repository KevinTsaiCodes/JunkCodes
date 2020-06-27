// app3_3, 印出Java定義的整數常數之最大值
public class app3_3
{
   public static void main(String args[])
   {
      long lmax=java.lang.Long.MAX_VALUE;
      int imax=java.lang.Integer.MAX_VALUE;
      short smax=Short.MAX_VALUE;   // 省略類別庫java.lang
      byte bmax=Byte.MAX_VALUE;  // 省略類別庫java.lang

      System.out.println("Max value of long  : "+lmax);
      System.out.println("Max value of int   : "+imax);
      System.out.println("Max value of short : "+smax);
      System.out.println("Max value of byte  : "+bmax);
   }
}