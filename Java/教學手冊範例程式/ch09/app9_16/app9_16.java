// app9_16, 在建構元裡建立內部類別的物件
public class app9_16
{
   public app9_16()
   {
      Caaa aa= new Caaa();
      aa.set_num(5);
   }

   public static void main(String args[])
   {
      app9_16 obj=new app9_16(); // 呼叫建構元app9_16()建立外部類別的物件
   }

   class Caaa
   {
      int num;
      void set_num(int n)
      {
         num=n;
         System.out.println("num= "+ num);
      }
   }
}

/* app9_16 OUTPUT---
num= 5
------------------*/