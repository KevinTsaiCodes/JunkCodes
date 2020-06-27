// app14_5, 利用FileInputStream讀取檔案
import java.io.*;
public class app14_5
{
   public static void main(String args[]) throws IOException
   {
      FileInputStream fi=new FileInputStream("c:\\Java\\train.txt");
      System.out.println("file size="+fi.available());
      byte ba[]=new byte[fi.available()]; // 建立byte陣列

      fi.read(ba);  // 將讀取的內容寫到陣列ba裡
      System.out.println(new String(ba)); // 印出陣列ba的內容
      fi.close();
   }
}

/* app14_5 OUTPUT----
file size=54
火車快飛，火車快飛
越過高山，飛過小溪
不知走了幾百里
--------------------*/