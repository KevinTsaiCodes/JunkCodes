// app11_4, 介面的實作範例
interface iShape2D               // 定義介面
{
   final double PI=3.14;
   abstract void area();
}

class CRectangle implements iShape2D // 以CRectangle類別實作iShape2D介面
{
   int width,height;
   public CRectangle(int w,int h)
   {
      width=w;
      height=h;
   }
   public void area()      // 定義area()的處理方式
   {
      System.out.println("area="+width*height);
   }
}

class CCircle implements iShape2D // 以CCircle類別實作iShape2D介面
{
   double radius;
   public CCircle(double r)
   {
      radius=r;
   }
   public void area()      // 定義area()的處理方式
   {
      System.out.println("area="+PI*radius*radius);
   }
}

public class app11_4
{
   public static void main(String args[])
   {
      CRectangle rect=new CRectangle(5,10);
      rect.area();         // 呼叫CRectangle類別裡的area() method

      CCircle cir=new CCircle(2.0);
      cir.area();       // 呼叫CCircle類別裡的area() method
   }
}

/* app11_4 OUTPUT---
area=50
area=12.56
------------------*/