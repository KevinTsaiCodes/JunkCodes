// app10_5, protected成員的使用
class CCircle
{
   protected static double pi=3.14;   // 將pi宣告成protected
   protected double radius;   // 將radius宣告成protected

   public void show()
   {
      System.out.println("area="+pi*radius*radius);
   }
}
class CCoin extends CCircle   // 定義CCoin類別，繼承自CCircle類別
{
   private int value;

   public CCoin(double r, int v)
   {
      radius=r;      // 在子類別裡可直接取用父類別裡的protected成員
      value=v;
      System.out.println("radius="+radius+", value="+value);
   }
}
public class app10_5
{
   public static void main(String args[])
   {
      CCoin coin=new CCoin(2.5,10);
      coin.show();
   }
}

/* app10_5 OUTPUT-------
radius=2.5, value=10
area=19.625
----------------------*/
