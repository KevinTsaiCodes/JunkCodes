// app9_1, 建構元的使用
class CCircle                 // 定義類別CCircle
{
   private double pi=3.14;
   private double radius;

   public CCircle(double r)      // 定義建構元CCircle()
   {
      radius=r;
   }
   public void show()
   {
       System.out.println("radius="+radius+", area="+pi*radius*radius);
   }
}
public class app9_1
{
   public static void main(String args[])
   {
      CCircle cir1=new CCircle(4.0); // 建立物件並呼叫CCircle()建構元
      cir1.show();
   }
}

/* app9_1 OUTPUT--------
radius=4.0, area=50.24
----------------------*/