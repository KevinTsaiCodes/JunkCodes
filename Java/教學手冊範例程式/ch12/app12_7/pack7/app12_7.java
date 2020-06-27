// app12_7.java, 此檔案置於pack7資料夾內
package pack7;             // 將app12_7類別納入package pack7當中
import pack7.subpack1.CCircle;   // 載入pack7.subpack1裡的CCircle類別
import pack7.subpack2.CBox;         // 載入pack7.subpack2裡的CBox類別

public class app12_7
{
   public static void main(String args[])
   {
      CCircle cir=new CCircle(2);
      CBox box=new CBox(2,3,4);
      cir.show();
      box.show();
   }
}

/* app12_7 OUTPUT--------------------------
radius=2.0, area=12.56
length=2, width=3, height=4, volume=24
-----------------------------------------*/