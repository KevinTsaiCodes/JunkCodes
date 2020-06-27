// app10_1, 簡單的繼承範例
class CCircle        // 父類別CCircle
{
   private static double pi=3.14;
   private double radius;

   public CCircle()     // CCircle()建構元
   {
      System.out.println("CCircle() constructor called ");
   }
   public void setRadius(double r)
   {
      radius=r;
      System.out.println("radius="+radius);
   }
   public void show()
   {
      System.out.println("area="+pi*radius*radius);
   }
}
class CCoin extends CCircle      // 子類別CCoin，繼承自CCircle類別
{
   private int value;            // 子類別的資料成員

   public CCoin()          // 子類別的建構元
   {
      System.out.println("CCoin() constructor called ");
   }
   public void setValue(int t)         // 子類別的setValue() method
   {
      value=t;
      System.out.println("value="+value);
   }
}
public class app10_1
{
   public static void main(String args[])
   {
      CCoin coin=new CCoin(); // 建立coin物件
      coin.setRadius(2.0);    // 呼叫由父類別繼承而來的setRadius()
      coin.show();         // 呼叫由父類別繼承而來的show()
      coin.setValue(5);       // 呼叫子類別的setValue()
   }
}

/* app10_1 OUTPUT----------------
CCircle() constructor called
CCoin() constructor called
radius=2.0
area=12.56
value=5
-------------------------------*/
