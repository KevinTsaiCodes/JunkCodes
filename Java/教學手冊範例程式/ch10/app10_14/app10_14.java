// app10_14, 改寫Object類別裡的toString() method
class Caaa
{
   private int num;

   public Caaa(int n)
   {
      num=n;
   }
   public String toString()    // 改寫toString() method
   {
      String str="toString() called, num="+num;
      return str;
   }
}
public class app10_14
{
   public static void main(String args[])
   {
      Caaa a=new Caaa(2);
      System.out.println(a.toString());   // 印出物件a的內容
   }
}

/* app10_14 OUTPUT---------
toString() called, num=2
-------------------------*/