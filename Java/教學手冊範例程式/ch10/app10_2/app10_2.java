// app10_2, 呼叫父類別中特定的建構元
class CCircle              // 定義父類別CCircle
{
   private static double pi=3.14;
   private double radius;

   public CCircle()        // 父類別裡沒有引數的建構元
   {
      System.out.println("CCircle() constructor called");
   }
   public CCircle(double r)   // 父類別裡有一個引數的建構元
   {
      System.out.println("CCircle(double r) constructor called");
      radius=r;
   }
   public void show()
   {
      System.out.println("area="+pi*radius*radius);
   }
}
class CCoin extends CCircle  // 定義子類別CCoin，繼承自CCircle類別
{
   private int value;
   public CCoin()                // 子類別裡沒有引數的建構元
   {
      System.out.println("CCoin() constructor called");
   }
   public CCoin(double r, int v) // 子類別裡有兩個引數的建構元
   {
      super(r);   // 呼叫父類別裡，有引數的建構元，即第11行所定義的建構元
      value=v;
      System.out.println("CCoin(double r, int v) constructor called");
   }
}
public class app10_2
{
   public static void main(String args[])
   {
      CCoin coin1=new CCoin();         // 建立物件，並呼叫第24行的建構元
      CCoin coin2=new CCoin(2.5,10);   // 建立物件，並呼叫第28行的建構元
      coin1.show();
      coin2.show();
   }
}

/* app10_2 OUTPUT--------------------------
CCircle() constructor called
CCoin() constructor called
CCircle(double r) constructor called
CCoin(double r, int v) constructor called
area=0.0
area=19.625
------------------------------------------*/

