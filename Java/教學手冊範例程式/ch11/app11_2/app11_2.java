// app11_2, 用抽象類別型態的變數來建立物件
abstract class CShape               // 定義抽象類別CShape
{
   protected String color;       // 資料成員
   public void setColor(String str)    // 一般的函數
   {
      color=str;
   }
   public abstract void show(); // 抽象函數，只有定義名稱，沒有定義處理方式
}



class CRectangle extends CShape    // 定義子類別CRectangle
{
   protected int width,height;
   public CRectangle(int w,int h)
   {
      width=w;
      height=h;
   }
   public void show()      // 明確定義繼承自抽象類別的show() method
   {
      System.out.print("color="+color+",  ");
      System.out.println("area="+width*height);
   }
}
class CCircle extends CShape     // 定義子類別CCircle
{
   protected double radius;
   public CCircle(double r)
   {
      radius=r;
   }
   public void show()     // 明確定義繼承自抽象類別的show() method
   {
      System.out.print("color="+color+",  ");
      System.out.println("area="+3.14*radius*radius);
   }
}
public class app11_2
{
   public static void main(String args[])
   {
      CShape shape1=new CRectangle(5,10);
      shape1.setColor("Yellow");
      shape1.show();

      CShape shape2=new CCircle(2.0);
      shape2.setColor("Green");
      shape2.show();
   }
}

/* app11_2 OUTPUT---------
color=Yellow,  area=50
color=Green,  area=12.56
------------------------*/