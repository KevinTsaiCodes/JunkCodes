// app12_6.java,此檔案置於pack6資料夾內
package pack6;  // 將app12_6類別納入package pack6當中
import pack6.subpack1.CCircle;   // 載入pack6.subpack1裡的CCircle類別
import pack6.subpack2.CRectangle;   // 載入pack6.subpack2裡的CRectangle類別

public class app12_6
{
   public static void main(String args[])
   {
      CCircle cir=new CCircle();
      CRectangle rect=new CRectangle();
      cir.show();
      rect.show();
   }
}
