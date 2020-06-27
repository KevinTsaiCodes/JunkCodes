// app17_4, AWT簡單的範例 (四)
import java.awt.*;
public class app17_4 extends Frame
{
   Label lab=new Label("Hello Java!!");   // 建立lab物件

   public app17_4(String str)             // 建構元app17_4()
   {
      super(str);             // 呼叫父類別(Frame)的建構元
      add(lab);               // 將標籤lab 物件加入視窗中
   }

   public static void main(String args[])
   {
      app17_4 frm=new app17_4("my first AWT program"); // 呼叫app17_4()建構元

      frm.setSize(220,150);
      frm.setBackground(Color.yellow);
      frm.setLocation(250,250);
      frm.setVisible(true);
   }
}
