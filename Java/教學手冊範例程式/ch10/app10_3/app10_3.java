// app10_3, 建構元錯誤的範例
class CCircle              // 定義類別CCircle
{
   private static double pi=3.14;
   private double radius;

   public CCircle(double r)   // 有引數的建構元
   {
      radius=r;
   }
   public void setRadius(double r)
   {
      radius=r;
      System.out.println("radius="+radius);
   }
}

class CCoin extends CCircle  // 定義CCoin類別，繼承自CCircle類別
{
   private int value;

   public CCoin(double r, int v)    // CCoin()有兩個引數的建構元
   {
      setRadius(r);        // 透過setRadius() method來設定radius成員
      value=v;             // 設定value成員
   }
}
public class app10_3
{
   public static void main(String args[])
   {
      CCoin coin1=new CCoin(2.5,10); // 建立物件，並呼叫有兩個引數的建構元
   }
}

/* app10_3 OUTPUT編譯器訊息---------------------------------------------------
error: constructor CCircle in class CCircle cannot be applied to given types;
   {
   ^
  required: double
  found: no arguments
  reason: actual and formal argument lists differ in length
----------------------------------------------------------------------------*/
