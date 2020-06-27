// app17_6, 在視窗中加入標籤物件
import java.awt.*;
public class app17_6
{
   static Frame frm=new Frame("Label class");
   static Label lab=new Label();          // 建立標籤物件lab

   public static void main(String args[])
   {
      frm.setSize(200,150);
      frm.setBackground(Color.pink);      // 設定視窗底色為粉紅色
      lab.setText("Hello Java");          // 在標籤內加上文字
      lab.setBackground(Color.white);     // 設定標籤底色為白色
      lab.setAlignment(Label.CENTER);        // 將標籤內的文字置中
      lab.setForeground(Color.blue);      // 設定標籤文字為藍色
      Font fnt=new Font("Serief",Font.ITALIC+Font.BOLD,18);
      lab.setFont(fnt);                // 設定字型的樣式
      frm.add(lab);
      frm.setVisible(true);
   }
}
