// app17_13, TextArea類別的應用
import java.awt.*;
public class app17_13
{
   static Frame frm=new Frame("TextArea class");
   static TextArea txa;

   public static void main(String args[])
   {
      txa=new TextArea("demo",8,14,TextArea.SCROLLBARS_VERTICAL_ONLY);
      frm.setLayout(null);             // 不使用版面配置
      txa.setBounds(30,45,140,80);        // 設定文字區的大小
      frm.setSize(200,150);
      frm.add(txa);
      frm.setVisible(true);
   }
}
