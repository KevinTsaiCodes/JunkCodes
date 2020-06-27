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

public class app11_5
{
   public static void main(String args[])
   {
      iShape2D var1,var2;  // 宣告介面型態的變數
      var1=new CRectangle(5,10); // 將介面型態的變數var1指向新建的物件
      var1.area();      // 透過介面var1呼叫show() method

      var2=new CCircle(2.0);  // 將介面型態的變數var2指向新建的物件
      var2.area();                  // 透過介面var2呼叫show() method
   }
}

/* app11_5 OUTPUT---
area=50
area=12.56
------------------*/