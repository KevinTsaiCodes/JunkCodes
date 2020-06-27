// app4_7,運算式的型態轉換
public class app4_7
{
   public static void main(String args[])
   {
      char ch='X';
      short s=-5;
      int i=6;
      float f=9.7f;
      double d=1.76;

      System.out.print("(s*ch)-(d/f)*(i+f)=");  // 印出結果
      System.out.println((s*ch)-(d/f)*(i+f));
   }
}

/* app4_7 OUTPUT-----------------------
(s*ch)-(d/f)*(i+f)=-442.8486598152212
-------------------------------------*/