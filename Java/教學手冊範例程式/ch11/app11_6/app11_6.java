// app11_6, 用CCircle類別實作兩個以上的介面
interface iShape2D         // 定義iShape2D介面
{
   final double PI=3.14;
   abstract void area();
}

interface iColor        // 定義iColor介面
{
   abstract void setColor(String str);
}

class CCircle implements iShape2D,iColor   // 實作iShape2D與iColor介面
{
   double radius;
   String color;
   public CCircle(double r)
   {
      radius=r;
   }
   public void setColor(String str)      // 定義iColor介面裡的setColor()
   {
      color=str;
      System.out.println("color="+color);
   }
   public void area()      // 定義iShape2D介面裡的area() method
   {
      System.out.println("area="+PI*radius*radius);
   }
}
public class app11_6
{
   public static void main(String args[])
   {
      CCircle cir;
      cir=new CCircle(2.0);
      cir.setColor("Blue");      // 呼叫setColor() method
      cir.area();          // 呼叫show() method
   }
}

/* app11_6 OUTPUT---
color=Blue
area=12.56
------------------*/