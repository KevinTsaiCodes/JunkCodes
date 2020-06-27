// app8_3, method的建立
class CRectangle
{
   int width;
   int height;
   int area()           // 定義成員函數area(), 用來計算面積
   {
      return width*height;       // 傳回矩形的面積
   }

   int perimeter()   // 定義成員函數perimeter(), 用來計算周長
   {
      return 2*(width+height);      // 傳回矩形的周長
   }
}

public class app8_3
{
   public static void main(String args[])
   {
      CRectangle rect1;
      rect1=new CRectangle();       // 建立新的物件

      rect1.width=20;            // 設定矩形rect1的寬
      rect1.height=15;           // 設定矩形rect1的高

      System.out.println("area="+rect1.area());
      System.out.println("perimeter="+rect1.perimeter());
   }
}

/* app8_3 OUTPUT---
area=300
perimeter=70
-----------------*/