// app8_2, 同時建立兩個物件
class CRectangle
{
   int width;         // 定義資料成員width
   int height;        // 定義資料成員height
}

public class app8_2
{
   public static void main(String args[])
   {
      CRectangle rect1,rect2;       // 宣告指變數rect1,rect2
      rect1=new CRectangle();       // 建立物件rect1
      rect2=new CRectangle();       // 建立物件rect2

      rect1.width=20;               // 設定矩形rect1的寬
      rect1.height=15;              // 設定矩形rect1的高

      rect2.width=25;               // 設定矩形rect2的寬
      rect2.height=rect1.height+3;     // 設定矩形rect2的高

      System.out.println("rect1.width="+rect1.width);
      System.out.println("rect1.height="+rect1.height);
      System.out.println("rect2.width="+rect2.width);
      System.out.println("rect2.height="+rect2.height);
   }
}

/* app8_2 OUTPUT---
rect1.width=20
rect1.height=15
rect2.width=25
rect2.height=18
-----------------*/