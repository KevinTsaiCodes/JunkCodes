// app19_3, 建立選擇表單
import java.awt.*;
public class app19_3 extends Frame
{
   static app19_3 frm=new app19_3();
   static Choice chc=new Choice();     // 建立下拉選單物件chc
   public static void main(String args[])
   {
      frm.setLayout(new FlowLayout(FlowLayout.CENTER,10,10));
      frm.setTitle("Choice class");
      for(int i=0;i<=4;i++)            // 利用for迴圈加入選項
         chc.add("Choice Item "+i);
      frm.add(chc);
      frm.setSize(200,150);
      frm.setBackground(Color.yellow);
      frm.setVisible(true);
   }
}
