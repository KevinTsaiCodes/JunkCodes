// app10_13, Object類別裡的toString() method
class Caaa
{
   private int num;

   public Caaa(int n)
   {
      num=n;
   }
}
public class app10_13
{
   public static void main(String args[])
   {
      Caaa a=new Caaa(2);
      System.out.println(a.toString());      // 印出物件a的內容
   }
}

/* app10_13 OUTPUT---
Caaa@757aef
-------------------*/