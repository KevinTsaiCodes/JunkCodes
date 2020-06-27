// app17_8, Button類別
import java.awt.*;
public class app17_8
{
   static Frame frm=new Frame("Button class");
   static Button btn=new Button("Push Me!!");      // 建立按鈕物件

   public static void main(String args[])
   {
      frm.setSize(200,150);
      frm.add(btn);                    // 在視窗內加入按鈕
      frm.setVisible(true);
   }
}
