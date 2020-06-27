// app8_7, 呼叫method並傳遞引數
class CCircle        //    類別CCircle
{
   double pi=3.14;    // 將資料成員設定初值
   double radius;

   void show_area()  // show_area() method, 顯示出半徑及圓面積
   {
      System.out.println("radius="+radius);
      System.out.println("area="+pi*radius*radius);
   }
   void setRadius(double r)   // setRadius() method, 可用來設定半徑
   {
      radius=r;               // 設定radius成員的值為r
   }
}
public class app8_7
{
   public static void main(String args[])
   {
      CCircle cir1=new CCircle();      // 宣告並建立新的物件
      cir1.setRadius(4.0);          // 設定cir1的半徑為4.0
      cir1.show_area();
   }
}

/* app8_7 OUTPUT---
radius=4.0
area=50.24
-----------------*/