// app14_1, 使用FileReader類別讀取檔案
import java.io.*;                // 載入java.io類別庫裡的所有類別
public class app14_1
{
   public static void main(String args[]) throws IOException
   {
      char data[]=new char[128]; // 建立可容納128個字元的陣列
      FileReader fr=new FileReader("c:\\Java\\train.txt");  // 建立物件fr

      int num=fr.read(data);     // 將資料讀入字元陣列data內
      String str=new String(data,0,num);     // 將字元陣列轉換成字串
      System.out.println("Characters read= "+num);
      System.out.println(str);

      fr.close();
   }
}

/* app14_1 OUTPUT------
Characters read= 29
火車快飛，火車快飛
越過高山，飛過小溪
不知走了幾百里
---------------------*/