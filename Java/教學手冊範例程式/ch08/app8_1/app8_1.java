// app8_1, 建立物件與field的存取
class CRectangle        // 定義CRectangle類別
{
   int width;           // 宣告資料成員width
   int height;             // 宣告資料成員height
}

public class app8_1
{
   public static void main(String args[])
   {
      CRectangle rect1;
      rect1=new CRectangle(); // 建立新的物件

      rect1.width=20;      // 設定矩形rect1的寬
      rect1.height=15;     // 設定矩形rect1的高

      System.out.println("width="+rect1.width);    // 印出rect1.width
      System.out.println("height="+rect1.height);    // 印出rect1.height
   }
}

/* app8_1 OUTPUT---
width=20
height=15
-----------------*/