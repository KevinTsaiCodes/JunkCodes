// app8_6, 在類別內部呼叫method
class CCircle        // 定義類別CCircle
{
  double pi=3.14;    // 將資料成員設定初值
  double radius;

  void show_area()      // show_area() method, 顯示出圓面積
  {
     System.out.println("area="+pi*radius*radius);
  }
  void show_all()       // show_all() method,同時顯示出半徑與圓面積
  {
     System.out.println("radius="+radius);
     show_area();          // 於類別內呼叫show_area() method
  }
}
public class app8_6
{
   public static void main(String args[])
   {
      CCircle cir1=new CCircle();   // 宣告並建立新的物件
      cir1.radius=2.0;
      cir1.show_all();     // 用cir1物件呼叫show_all()
   }
}

/* app8_6 OUTPUT---
radius=2.0
area=12.56
-----------------*/