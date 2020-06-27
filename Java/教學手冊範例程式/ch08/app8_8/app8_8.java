// app8_8, 圓形類別CCircle
class CCircle        // 定義類別CCircle
{
   double pi;        // 將資料成員設定初值
   double radius;

   void show_area()        // show_area() method, 顯示出圓面積
   {
      System.out.println("area="+pi*radius*radius);
   }
   void setCircle(double p,double r)   // 擁有兩個引數的method
   {
      pi=p;
      radius=r;
   }
}
public class app8_8
{
   public static void main(String args[])
   {
      CCircle cir1=new CCircle();   // 宣告並建立新的物件
      cir1.setCircle(3.1416,2.0);   // 呼叫並傳遞引數到setCircle()
      cir1.show_area();
   }
}

/* app8_8 OUTPUT---
width=20
height=15
-----------------*/