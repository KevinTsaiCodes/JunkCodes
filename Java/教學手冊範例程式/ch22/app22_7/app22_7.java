// app22_7, 建立Client端的伺服程式
import java.net.*;
import java.io.*;

public class app22_7
{
   public static void main(String args[])
   {
      byte buff[]=new byte[1024];         // 建立byte型態的陣列
      try
      {
         System.out.println("正在與伺服器建立連線...");
         Socket s=new Socket("127.0.0.1",2525);    // 建立socket物件
         System.out.println("已經與伺服器取得連線...");
         InputStream in=s.getInputStream();     // 建立輸入串流
         int n=in.read(buff); // 從串流讀入資料
         System.out.print("從伺服端收到: ");
         System.out.println (new String(buff,0,n));// 印出讀入的內容
         in.close();
         s.close();
      }
      catch(Exception e)
      {
         System.out.println("發生了"+e+"例外");
      }
   }
}