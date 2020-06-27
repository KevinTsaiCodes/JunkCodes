// app10_12, 利用equals()來判別兩個類別變數所指向的是否為同一個物件
class Caaa              // 定義Caaa類別
{
   private int num;

   public Caaa(int n)
   {
      num=n;
   }
}
public class app10_12
{
   public static void main(String args[])
   {
      Caaa a=new Caaa(5);
      Caaa b=new Caaa(5);
      Caaa c=a;          // 宣告類別變數c，並讓它指向變數a所指向的物件
      boolean br1=a.equals(b);   // 測試a與b是否指向同一物件
      boolean br2=a.equals(c);   // 測試a與c是否指向同一物件
      System.out.println("a.equals(b)="+br1);
      System.out.println("a.equals(c)="+br2);
   }
}

/* app10_12 OUTPUT---
a.equals(b)=false
a.equals(c)=true
-------------------*/
