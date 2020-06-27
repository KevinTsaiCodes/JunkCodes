// app3_7, 列印跳脫序列
public class app3_7
{
   public static void main(String args[])
   {
      char ch1='\"';          // 將ch1設值為\"
      char ch2='\74';         // 以八進位值設定字元變數ch2
      char ch3='\u003e';      // 以16進位值設定字元變數ch3

      System.out.println(ch1+"Time flies."+ch1);
      System.out.println("\"Time is money!\"");
      System.out.println(ch2+"Tomorrow never comes"+ch3);
   }
}

/* app3_7 OUTPUT------------
"Time flies."
"Time is money!"
<Tomorrow never comes>
---------------------------*/