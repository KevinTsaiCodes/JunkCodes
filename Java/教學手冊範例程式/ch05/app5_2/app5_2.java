// app5_2, 條件運算子?:的使用
public class app5_2
{
   public static void main(String args[])
   {
      int a=8,b=3,max;

      max=(a>b)?a:b;          // a>b時,max=a,否則max=b

      System.out.println("a="+a+", b="+b);
      System.out.println(max+"是較大的數");
   }
}

/* app5_2 OUTPUT------
a=8, b=3
8是較大的數
--------------------*/