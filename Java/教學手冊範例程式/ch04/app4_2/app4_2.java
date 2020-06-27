// app4_2, 一元運算子「~」與「!」
public class app4_2
{
   public static void main(String args[])
   {
      byte a=Byte.MIN_VALUE;  // 宣告變數a,並設為該型態之最小值
      boolean b=true;         // 宣告boolean變數b,並設為true

      System.out.println("a="+a+",~a="+(~a));  // 印出a與~a的值
      System.out.println("b="+b+",!b="+(!b));  // 印出b與!b的值
   }
}