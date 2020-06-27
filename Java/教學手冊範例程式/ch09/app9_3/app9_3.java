// app9_3, 從某一建構元呼叫另一建構元
class CCircle              // 定義類別CCircle
{
   private String color;
   private double pi=3.14;
   private double radius;

   public CCircle()           // 沒有引數的建構元
   {
      this("Green",1.0);      // 此行會呼叫第13行的建構元
      System.out.println("constructor CCircle() called");
   }
   public CCircle(String str, double r)    // 有引數的建構元
   {
      System.out.println("constructor CCircle(String,double) called");
      color=str;
      radius=r;
   }
   public void show()
   {
      System.out.println("color="+color+", Radius="+radius);
      System.out.println("area="+pi*radius*radius);
   }
}
public class app9_3
{
   public static void main(String args[])
   {
      CCircle cir1=new CCircle();
      cir1.show();
   }
}

/* app9_3 OUTPUT----------------------------
constructor CCircle(String,double) called
constructor CCircle() called
color=Green, Radius=1.0
area=3.14
------------------------------------------*/