// app8_9, 圓形類別CCircle
class CCircle           // 定義類別CCircle
{
   double pi;           // 將資料成員設定初值
   double radius;

   double getRadius()      // getRadius(), 用來傳回物件的半徑
   {
      return radius;
   }
   void setCircle( double p, double r)
   {
      pi=p;
      radius=r;
   }
}
public class app8_9
{
   public static void main(String args[])
   {
      CCircle cir1=new CCircle();      // 宣告並建立新的物件
      cir1.setCircle(3.1416,2.0);
      System.out.println("radius="+cir1.getRadius());
   }
}

/* app8_9 OUTPUT---
radius=2.0
-----------------*/