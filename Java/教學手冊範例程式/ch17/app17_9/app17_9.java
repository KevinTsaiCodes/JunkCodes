// app17_9, 設定按鈕的大小
import java.awt.*;
public class app17_9
{
   static Frame frm=new Frame("Button class");
   static Button btn=new Button("Push Me!!");

   public static void main(String args[])
   {
      frm.setLayout(null);          // 不使用版面配置
      btn.setBounds(45,65,100,40);     // 設定按鈕的大小與位置
      frm.setSize(200,150);
      frm.add(btn);
      frm.setVisible(true);
   }
}
