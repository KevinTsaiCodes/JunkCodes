// app11_7, 介面的延伸
interface iShape              // 定義iShape介面
{
   final double PI=3.14;
   abstract void setColor(String str);
}

interface iShape2D extends iShape  // 定義iShape2D介面, 繼承自iShape
{
   abstract void area();
}

class CCircle implements iShape2D  // 實作iShape2D介面
{
   double radius;
   String color;

   public CCircle(double r)
   {
      radius=r;
   }
   public void setColor(String str)    // 定義iShape介面的setColor()
   {
      color=str;
      System.out.println("color="+color);
   }
   public void area()                  // 定義iShape2D介面裡的area()
   {
      System.out.println("area="+PI*radius*radius);
   }
}
public class app11_7
{
   public static void main(String args[])
   {
      CCircle cir;
      cir=new CCircle(2.0);
      cir.setColor("Blue");         // 呼叫setColor() method
      cir.area();                // 呼叫area() method
   }
}

/* app11_7 OUTPUT---
color=Blue
area=12.56
------------------*/