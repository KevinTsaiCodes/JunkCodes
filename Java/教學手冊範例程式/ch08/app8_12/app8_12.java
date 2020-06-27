// app8_12, 圓形類別CCircle
class CCircle           // 定義類別CCircle
{
   double pi=3.14;      // 將資料成員設定初值
   double radius;

   void show_area()
   {
      System.out.println("area="+pi*radius*radius);
   }
}

public class app8_12
{
   public static void main(String args[])
   {
      CCircle cir1=new CCircle();
      cir1.radius=-2.0;
      cir1.show_area();
   }
}

/* app8_12 OUTPUT---
area=12.56
------------------*/