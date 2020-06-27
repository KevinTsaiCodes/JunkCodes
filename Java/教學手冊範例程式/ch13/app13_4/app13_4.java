// app13_4, 於程式中拋出例外
public class app13_4
{
   public static void main(String args[])
   {
      int a=4,b=0;

      try
      {
        if(b==0)
         throw new ArithmeticException();    // 拋出例外
        else
         System.out.println(a+"/"+b+"="+a/b);  // 若沒有拋出例外，則執行此行
      }
      catch(ArithmeticException e)
      {
         System.out.println(e+" throwed");
      }
   }
}

/* app13_4 OUTPUT-----------------------
java.lang.ArithmeticException throwed
--------------------------------------*/