// app14_3, 從緩衝區裡讀入資料
import java.io.*;
public class app14_3
{
   public static void main(String args[]) throws IOException
   {
      String str;
      int count=0;
      FileReader fr=new FileReader("c:\\Java\\number.txt");
      BufferedReader bfr=new BufferedReader(fr);

      while((str=bfr.readLine())!=null)   // 每次讀取一行，直到檔案結束
      {
         count++;                   // 計算讀取的行數
         System.out.println(str);
      }
      System.out.println(count+" lines read");

      fr.close();                   // 關閉檔案
   }
}

/* app14_3 OUTPUT---
12
34
63
14
16
56
6 lines read
-------------------*/