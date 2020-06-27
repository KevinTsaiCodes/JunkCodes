// app10_11, 利用getClass()取得呼叫物件所屬的類別
class Caaa                    // 定義Caaa類別
{
   private int num;

   public Caaa(int n)
   {
      num=n;
   }
}
public class app10_11
{
   public static void main(String args[])
   {
      Caaa a=new Caaa(5);
      Class ca=a.getClass();        // 用變數a呼叫getClass()
      System.out.println("Class of obj = "+ca);
   }
}

/* app10_11 OUTPUT----------
Class of obj = class Caaa
--------------------------*/

