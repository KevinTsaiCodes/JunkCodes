// app13_8, 定義自己的例外類別
class CCircleException extends Exception  // 定義自己的例外類別
{
}

class CCircle        // 定義類別CCircle
{
   private double radius;
   public void setRadius(double r) throws CCircleException
   {
      if(r<0)
      {
         throw new CCircleException();       // 拋出例外
      }
      else
         radius=r;
   }

   public void show()
   {
      System.out.println("area="+3.14*radius*radius);
   }
}

public class app13_8
{
   public static void main(String args[])
   {
      CCircle cir=new CCircle();
      try
      {
         cir.setRadius(-2.0);
      }
      catch(CCircleException e)    // 捕捉由setRadius()拋出的例外
      {
         System.out.println(e+" throwed");
      }
      cir.show();
   }
}

/* app13_8 OUTPUT-----------
CCircleException throwed
area=0.0
--------------------------*/