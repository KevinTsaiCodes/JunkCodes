// app8_14, 公有成員(method)的建立
class CCircle              // 定義類別CCircle
{
   private double pi=3.14;       // 將資料成員設定為private
   private double radius;

   private double area()      // 定義私有的成員函數area()
   {
       return pi*radius*radius;
   }
   public void show_area()    // 定義公有的成員函數show_area()
   {
       System.out.println("area="+ area());  // 呼叫私有成員area()
   }
   public void setRadius(double r)  // 定義公有的成員函數setRadius()
   {
       if(r>0)
       {
          radius=r;           // 將私有成員radius設為r
          System.out.println("radius="+radius);
       }
       else
          System.out.println("input error");
   }
}

public class app8_14
{
   public static void main(String args[])
   {
       CCircle cir1=new CCircle();
       cir1.setRadius(-2.0);  // 呼叫公有的setRadius() method
       cir1.show_area();         // 呼叫公有的show_area() method
   }
}

/* app8_14 OUTPUT---
input error
area=0.0
------------------*/