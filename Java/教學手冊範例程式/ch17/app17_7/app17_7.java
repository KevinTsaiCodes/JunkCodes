// app17_7, 指定標籤物件的大小
import java.awt.*;
public class app17_7
{
   static Frame frm=new Frame("Label class");
   static Label lab=new Label();

   public static void main(String args[])
   {
      frm.setLayout(null);             // 取消版面配置
      frm.setSize(200,150);
      frm.setBackground(Color.pink);
      lab.setText("Hello Java");
      lab.setBackground(Color.white);
      lab.setAlignment(Label.CENTER);
      lab.setForeground(Color.blue);
      lab.setLocation(60,50);          // 設定標籤位置
      lab.setSize(120,20);             // 設定標籤大小
      lab.setFont(new Font("Serief",Font.ITALIC+Font.BOLD,18));
      frm.add(lab);
      frm.setVisible(true);
   }
}
