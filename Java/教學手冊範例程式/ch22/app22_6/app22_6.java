// app22_6, 建立Server端的伺服程式
import java.net.*;
import java.io.*;

public class app22_6
{
   public static void main(String args[])
   {
      try
      {
         ServerSocket svs=new ServerSocket(2525);

         System.out.println("等候客戶端的請求中...");
         Socket s=svs.accept();   // 等候客戶端的請求
         System.out.println("客戶端已和本機連上線...");

         OutputStream out=s.getOutputStream();   // 取得輸出串流
         String str="Honor shows the man.";
         System.out.println("資料正在傳送中...");
         out.write(str.getBytes()); // 將字串轉成Byte陣列，再寫入串流中
         out.close();   // 關閉輸出串流
         s.close();  // 關閉socket
         System.out.println("資料傳送完畢...");
      }
      catch(Exception e)
      {
         System.out.println("發生了"+e+"例外");
      }
   }
}