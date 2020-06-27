// app17_3, AWT簡單的範例 (三)
import java.awt.*;
public class app17_3 extends Frame  // 指定app17_3繼承自Frame類別
{
   public static void main(String args[])
   {
      app17_3 frm=new app17_3();     // 用app17_3類別產生frm物件

      Label lab=new Label("Hello Java!!");
      frm.setTitle("my first AWT program");      // 在視窗中加入標題
      frm.setSize(220,150);
      frm.setBackground(Color.yellow);
      frm.setLocation(250,250);
      frm.add(lab);
      frm.setVisible(true);
   }
}
