// app9_5, 簡單的範例:實例變數與實例函數
class CCircle              // 定義類別CCircle
{
   private double pi=3.14;
   private double radius;

   public CCircle(double r)   // CCircle()建構元
   {
      radius=r;
   }
   public void show()
   {
      System.out.println("area="+pi*radius*radius);
   }
}
public class app9_5
{
   public static void main(String args[])
   {
      CCircle cir1=new CCircle(1.0);
      cir1.show();         // show()必須透過物件來呼叫
      CCircle cir2=new CCircle(2.0);
      cir2.show();         // show()必須透過物件來呼叫
   }
}

/* app9_5 OUTPUT---
area=3.14
area=12.56
-----------------*/
