// app8_13, 私有成員無法從類別外部來存取的範例
class CCircle              // 設定field為私有成員
{
   private double pi=3.14;       // 將資料成員設定初值
   private double radius;

   void show_area()
   {
      System.out.println("area="+pi*radius*radius);
   }
}

public class app8_13
{
   public static void main(String args[])
   {
      CCircle cir1=new CCircle();
      cir1.radius=-2.0;
      cir1.show_area();
   }
}

/* app8_13 OUTPUT---
width=20
height=15
------------------*/