// app8_10, 函數的多載(一)
class CCircle                    // 定義類別CCircle
{
   String color;
   double pi=3.14;
   double radius;

   void setColor(String str)        // 設定color的method
   {
      color=str;
   }
   void setRadius(double r)         // 設定radius的method
   {
      radius=r;
   }
   void setAll(String str, double r)   // 同時設定color與radius
   {
      color=str;
      radius=r;
   }
   void show()                   // 列印半徑、顏色與圓面積
   {
      System.out.println("color="+color+", Radius="+radius);
      System.out.println("area="+pi*radius*radius);
   }
}
public class app8_10
{
   public static void main(String args[])
   {
      CCircle cir1=new CCircle();

      cir1.setColor("Red");      // 設定cir1的color
      cir1.setRadius(2.0);       // 設定cir1的radius
      cir1.show();

      cir1.setAll("Blue",4.0);   // 同時設定cir1的color和radius
      cir1.show();
   }
}