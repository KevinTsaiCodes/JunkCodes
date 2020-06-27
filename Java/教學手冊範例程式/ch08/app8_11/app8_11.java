// app8_11, 函數的多載(二)
class CCircle                    // 定義類別CCircle
{
   String color;
   double pi=3.14;
   double radius;

   void setCircle(String str)          // 設定color成員
   {
      color=str;
   }
   void setCircle(double r)         // 設定radius成員
   {
      radius=r;
   }
   void setCircle(String str, double r)   // 同時設定color與radius
   {
      color=str;
      radius=r;
   }
   void show()
   {
      System.out.println("color="+color+", Radius="+radius);
      System.out.println("area="+pi*radius*radius);
   }
}

public class app8_11
{
   public static void main(String args[])
   {
      CCircle cir1=new CCircle();

      cir1.setCircle("Red");     // 呼叫第8行的setCircle()
      cir1.setCircle(2.0);       // 呼叫第12行的setCircle()
      cir1.show();

      cir1.setCircle("Blue",4.0);   // 呼叫第16行的setCircle()
      cir1.show();
   }
}

/* app8_11 OUTPUT------
color=Red, Radius=2.0
area=12.56
color=Blue, Radius=4.0
area=50.24
---------------------*/