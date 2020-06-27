// app9_15, 內部類別的撰寫
public class app9_15
{
   public static void main(String args[])
   {
      Caaa aa= new Caaa();
      aa.set_num(5);
   }

   static class Caaa
   {
      int num;
      void set_num(int n)
      {
         num=n;
         System.out.println("num= "+ num);
      }
   }
}

/* app9_15 OUTPUT---
num= 5
------------------*/