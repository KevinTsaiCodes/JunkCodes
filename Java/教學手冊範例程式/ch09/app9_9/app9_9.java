// app9_9, 類別型態之變數的應用
class CCircle              // 定義類別CCircle
{
   private static double pi=3.14;
   private double radius;

   public CCircle(double r)   // CCircle建構元
   {
      radius=r;
   }
   public void setRadius(double r)
   {
      radius=r;               // 設定radius成員的值
   }
   public void show()
   {
      System.out.println("area="+pi*radius*radius);
   }
}
public class app9_9
{
   public static void main(String args[])
   {
      CCircle cir1,cir2;
      cir1=new CCircle(1.0);
      cir1.show();

      cir2=cir1; // 將cir1設給cir2，此時這兩個變數所指向的內容均相等
      cir2.setRadius(2.0);  // 將cir2物件的半徑設為2.0
      cir1.show();
   }
}

/* app9_9 OUTPUT---
area=3.14
area=12.56
-----------------*/