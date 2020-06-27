// app17_2, AWT簡單的範例 (二)
import java.awt.*;            // 載入java.awt類別庫裡的所有類別
public class app17_2
{
   public static void main(String args[])
   {
      Frame frm=new Frame("my first AWT program");
      Label lab=new Label("Hello Java!!");
      frm.setSize(220,150);
      frm.setBackground(Color.yellow);
      frm.setLocation(250,250);
      frm.add(lab);
      frm.setVisible(true);
   }
}
