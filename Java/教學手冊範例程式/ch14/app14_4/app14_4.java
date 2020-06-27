// app14_4, 將資料寫到緩衝區內
import java.io.*;
public class app14_4
{
   public static void main(String args[]) throws IOException
   {
      FileWriter fw=new FileWriter("c:\\Java\\random.txt");
      BufferedWriter bfw=new BufferedWriter(fw);

      for(int i=1;i<=5;i++)
      {
         bfw.write(Double.toString(Math.random()));  // 寫入亂數到緩衝區
         bfw.newLine(); // 寫入換行符號
      }
      bfw.flush();      // 將緩衝區內的資料寫到檔案裡
      fw.close();       // 關閉檔案
   }
}