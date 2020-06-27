// app12_5.java, package的使用(四),此檔案置於pack5a資料夾內
package pack5a;
import pack5b.CCircle;   // 載入pack5b package裡的CCircle類別

public class app12_5
{
   public static void main(String args[])
   {
      CCircle cir=new CCircle();    // 不用再寫package的名稱
      cir.show();
   }
}
