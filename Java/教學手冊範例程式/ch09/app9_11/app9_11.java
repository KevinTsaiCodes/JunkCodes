// app9_11, 由method傳回類別型態的變數
class CCircle              // 定義類別CCircle
{
   private static double pi=3.14;
   private double radius;

   public CCircle(double r)   // CCircle建構元
   {
      radius=r;
   }
   public CCircle compare(CCircle cir) // Compare() method
   {
      if(this.radius>cir.radius)
         return this;         // 傳回呼叫compare() method的物件
      else
         return cir;          // 傳回傳入compare() method的物件
   }
}
public class app9_11
{
   public static void main(String args[])
   {
      CCircle cir1=new CCircle(1.0);
      CCircle cir2=new CCircle(2.0);
      CCircle obj;

      obj=cir1.compare(cir2);      // 呼叫compare() method
      if(cir1==obj)
         System.out.println("radius of cir1 is larger");
      else
         System.out.println("radius of cir2 is larger");
   }
}

/* app9_11 OUTPUT----------
radius of cir2 is larger
-------------------------*/