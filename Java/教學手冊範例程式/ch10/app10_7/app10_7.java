// app10_7, 透過父類別變數cir呼叫show() method
class CCircle           // 父類別CCircle
{
   protected static double pi=3.14;
   protected double radius;

   public CCircle(double r)
   {
      radius=r;
   }
   public void show()         // 父類別裡的show() method
   {
      System.out.println("radius="+radius);
   }
}
class CCoin extends CCircle   // 子類別CCircle
{
   private int value;

   public CCoin(double r,int v)
   {
      super(r);
      value=v;
   }
   public void show()       // 子類別裡的show() method
   {
      System.out.println("radius="+radius+", value="+value);
   }
   public void showValue()   // showValue() method,此函數只存在於子類別
   {
      System.out.println("value="+value);
   }
}
public class app10_7
{
   public static void main(String args[])
   {
      CCircle cir=new CCoin(2.0,5); // 宣告父類別變數cir，並將它指向物件
      cir.show();                   // 利用父類別變數cir呼叫show()
      // cir.showValue();
   }
}

/* app10_7 OUTPUT------
radius=2.0, value=5
---------------------*/
