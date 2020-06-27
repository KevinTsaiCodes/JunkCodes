// app12_8.java,此檔案置於pack8資料夾內
package pack8;  // 將app12_8類別納入package pack8當中
import pack8.subpack1.CCircle;     // 載入pack8.subpack1裡的CCircle類別
import pack8.subpack2.CBox;           // 載入pack8.subpack2裡的CBox類別
//import pack8.subpack3.CRectangle; // 載入pack8.subpack3裡的CRectangle類別

 public class app12_8
 {
    public static void main(String args[])
    {
       CCircle cir=new CCircle(2);
       CBox box=new CBox(2,3,4);
       // CRectangle rec=new CRectangle(5,7);
       cir.show();
       box.show();
       // rec.show();
    }
 }
