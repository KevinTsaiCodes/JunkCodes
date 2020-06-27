// app4_4, 遞增運算子「++」
public class app4_4
{
   public static void main(String args[])
   {
      int a=5,b=5,c;

      System.out.println("a="+a+", b="+b);  // 印出a、b
      c=a++;
      System.out.println("c="+c+", a="+a);  // 印出a++
      c=++b;
      System.out.println("c="+c+", b="+b);  // 印出++b
   }
}

/* app4_4 OUTPUT------------
a=5, b=5
c=5, a=6
c=6, b=6
---------------------------*/