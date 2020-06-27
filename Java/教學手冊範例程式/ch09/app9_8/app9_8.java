// app9_8, 設值給類別型態的變數
class CCircle        // 定義類別CCircle
{
   private static double pi=3.14;
   private double radius;

   public CCircle(double r)
   {
       radius=r;
   }
   public void show()
   {
       System.out.println("area="+pi*radius*radius);
   }
}
public class app9_8
{
   public static void main(String args[])
   {
      CCircle cir1,cir2;      // 宣告cir1,cir2為類別型態的變數
      cir1=new CCircle(1.0);  // 建立新的物件，並將cir1指向它
      cir1.show();

      cir2=cir1;    // 將cir1設給cir2，此時這兩個變數所指向的內容均相等
      cir2.show();

      CCircle cir3=new CCircle(2.0); // 建立新的物件，並將cir3指向它
      cir3.show();
   }
}

/* app9_8 OUTPUT---
area=3.14
area=3.14
area=12.56
-----------------*/