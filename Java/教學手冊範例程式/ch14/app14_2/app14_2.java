// app14_2, 使用FileWriter類別將資料寫入檔案內
import java.io.*;
public class app14_2
{
   public static void main(String args[]) throws IOException
   {
      FileWriter fw=new FileWriter("c:\\Java\\proverb.txt");
      char data[]={'T','i','m','e',' ','f','l','i','e','s','!','\r','\n'};
      String str="End of file";
      fw.write(data);         // 將字元陣列寫到檔案裡
      fw.write(str);          // 將字串寫到檔案裡
      fw.close();
   }
}