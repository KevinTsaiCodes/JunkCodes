// app13_6, 指定method拋出例外
public class app13_6
{
   public static void aaa(int a,int b) throws ArithmeticException
   {
      int c;
      c=a/b;
      System.out.println(a+"/"+b+"="+c);
   }

   public static void main(String args[])
   {
      try
      {
         aaa(4,0);
      }
      catch(ArithmeticException e)
      {
         System.out.println(e+" throwed");
      }
   }
}

/* app13_6 OUTPUT----------------------------------
java.lang.ArithmeticException: / by zero throwed
-------------------------------------------------*/