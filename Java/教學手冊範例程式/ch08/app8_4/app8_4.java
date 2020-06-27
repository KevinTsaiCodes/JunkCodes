// app8_4, 圓形類別CCircle
class CCircle           // 定義類別CCircle
{
   double pi=3.14;      // 將資料成員設定初值
   double radius;

   void show_area()     // show_area() method, 顯示出圓面積
   {
      System.out.println("area="+pi*radius*radius);
   }
}
public class app8_4
{
   public static void main(String args[])
   {
      CCircle cir1=new CCircle();   // 建立cirl物件
      cir1.radius=2.0;        // 設定radius的值
      cir1.show_area();       // 呼叫show_area() method
   }
}

/* app8_4 OUTPUT---
area=12.56
-----------------*/