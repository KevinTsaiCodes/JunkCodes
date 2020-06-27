// app9_6,「類別變數」的使用
class CCircle                    // 定義類別CCircle
{
   private static int count=0;         // 宣告count為「類別變數」
   private static double pi=3.14;      // 宣告pi為「類別變數」
   private double radius;

   public CCircle()        // 沒有引數的CCircle()建構元
   {
      this(1.0);           // 呼叫第12行的建構元，並傳入1.0
   }
   public CCircle(double r)   // 有一個引數的CCircle()建構元
   {
      radius=r;
      count ++;               // 當此建構元被呼叫時，count便加1
   }
   public void show()
   {
      System.out.println("area="+pi*radius*radius);
   }
   public void show_count()   // show_count(),顯示目前物件建立的個數
   {
      System.out.println(count+" object(s) created");
   }
}
public class app9_6
{
   public static void main(String args[])
   {
      CCircle cir1=new CCircle();      // 呼叫第8行的建構元
      cir1.show_count();     // 用cir1物件呼叫show_count() method
      CCircle cir2=new CCircle(2.0);   // 呼叫第12行的建構元
      CCircle cir3=new CCircle(4.3);   // 呼叫第12行的建構元
      cir1.show_count();     // 用cir1物件呼叫show_count() method
      cir2.show_count();     // 改用cir2物件呼叫show_count() method
      cir3.show_count();     // 改用cir3物件呼叫show_count() method
   }
}