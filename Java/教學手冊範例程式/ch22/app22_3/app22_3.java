// app22_3, 使用URL類別
import java.net.*;

public class app22_3
{
   public static void main(String args[])
   {
      try
      {
         URL u=new URL("http://udn.com/NEWS/main.html");

         System.out.println("通訊協定名稱為 "+u.getProtocol());
         System.out.println("host name為 "+u.getHost());
         System.out.println("埠號為 "+u.getPort());
         System.out.println("檔名為 "+u.getFile());
      }
      catch(MalformedURLException e)
      {
         System.out.println("發生了" +e+ "例外");
      }
   }
}

/* app22_3 OUTPUT--------
通訊協定名稱為 http
host name為 udn.com
埠號為 -1
檔名為 /NEWS/main.html
-----------------------*/
