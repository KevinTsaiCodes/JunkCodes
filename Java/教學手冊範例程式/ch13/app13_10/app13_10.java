// app13_10, 捕捉IOException例外
import java.io.*;             // 載入java.io類別庫裡的所有類別
public class app13_10
{
   public static void main(String args[])
   {
      BufferedReader buf;
      String str;

      buf=new BufferedReader(new InputStreamReader(System.in));
      try
      {
         System.out.print("Input a string: ");
         str=buf.readLine();
         System.out.println("string= "+str);       // 印出字串
      }
      catch(IOException e){}
   }
}

/* app13_10 OUTPUT-------------
Input a string: Hello Java!
string= Hello Java!
----------------------------*/