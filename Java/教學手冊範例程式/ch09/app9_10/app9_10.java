// app9_10, 傳遞類別型態的變數
class CCircle
{
   private static double pi=3.14;
   private double radius;

   public CCircle(double r)        // CCircle()建構元
   {
      radius=r;
   }
   public void compare(CCircle cir) // compare() method
   {
      if(this.radius==cir.radius)    // 判別物件的radius成員是否相等
         System.out.println("radius are equal");
      else
         System.out.println("radius are not equal");
   }
}
public class app9_10
{
   public static void main(String args[])
   {
      CCircle cir1=new CCircle(1.0);
      CCircle cir2=new CCircle(2.0);
      cir1.compare(cir2);     // 比較cir1與cir2的radius是否相等
   }
}

/* app9_10 OUTPUT----
radius are not equal
-------------------*/