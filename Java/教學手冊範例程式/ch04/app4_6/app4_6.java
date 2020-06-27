// app4_6, 簡潔運算式
public class app4_6
{
   public static void main(String args[])
   {
      int a=2,b=5;

      System.out.println("計算a+=b前, a="+a+" ,b="+b);
      a+=b;       // 計算a+=b的值，此式相同於a=a+b
      System.out.println("計算a+=b後, a="+a+" ,b="+b);
   }
}

/* app4_6 OUTPUT------------
計算a+=b前, a=2 ,b=5
計算a+=b後, a=7 ,b=5
---------------------------*/