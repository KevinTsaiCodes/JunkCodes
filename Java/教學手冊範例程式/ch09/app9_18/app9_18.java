// app9_18, 匿名內部類別
public class app9_18
{
   public static void main(String args[])
   {
      (new Caaa(){void set_num(int n){num=n;
       System.out.println("num= "+ num);}}).set_num(5);
   }
   static class Caaa
   {
       int num;
   }
}

/* app9_18 OUTPUT---
num= 5
------------------*/
