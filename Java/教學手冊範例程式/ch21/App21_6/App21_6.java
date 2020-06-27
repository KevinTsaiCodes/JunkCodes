// App21_6, 加入AWT元件到applet裡
import java.awt.*;
import java.awt.event.*;
import java.applet.Applet;

public class App21_6 extends Applet implements ActionListener
{
   Button btn;                // 宣告Button型態的變數btn
   public void init()
   {
        btn=new Button("Start");    // 建立btn物件
        btn.addActionListener(this);   // 以applet本身當成btn的傾聽者
        add(btn);  // 將btn按鈕加入applet視窗裡
   }
   public void actionPerformed(ActionEvent e)
   {
      if(btn.getLabel()=="Start")
           btn.setLabel("Stop"); // 設定按鈕上方的文字為Stop
      else
           btn.setLabel("Start");   // 設定按鈕上方的文字為Start
   }
}
