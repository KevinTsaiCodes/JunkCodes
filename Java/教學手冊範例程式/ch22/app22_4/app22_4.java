// app22_4, 載入URL的檔案內容
import java.net.*;
import java.io.*;

public class app22_4
{
   public static void main(String args[])
   {
      String str;

      try
      {
         URL u=new URL("file:/c:\\java\\poem.txt");

         Object obj=u.getContent();             // 取得URL的內容
         InputStreamReader isr=new InputStreamReader((InputStream) obj);
         BufferedReader br=new BufferedReader(isr);

         while((str=br.readLine())!=null)
            System.out.println(str);
         br.close();
      }
      catch(IOException e)
      {
         System.out.println("發生了"+e+"例外");
      }
   }
}

/* app22_4 OUTPUT---
床前明月光，
疑是地上霜。
舉頭望明月，
低頭思故鄉。
-------------------*/