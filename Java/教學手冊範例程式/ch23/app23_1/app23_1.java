// app23_1, JFrame類別的練習
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;         // 載入javax.swing類別庫裡的所有類別

public class app23_1 extends JFrame implements ActionListener
{
   static app23_1 frm=new app23_1();
   static Button btn=new Button("Click Me");
   static Container cp=frm.getContentPane();       // 取得視窗容器

   public static void main(String args[])
   {
       cp.add(btn);                    // 將按鈕btn加入內容層中
       cp.setLayout(new FlowLayout());    // 設定內容層的版面配置
       cp.setBackground(Color.pink);      // 設定內容層的顏色
       btn.addActionListener(frm);
       frm.setTitle("JFrame 視窗");
       frm.setSize(200,150);
       frm.setVisible(true);
   }
   // 按下btn按鈕的事件處理
   public void actionPerformed(ActionEvent e)
   {
       if(cp.getBackground()==Color.pink)
        cp.setBackground(Color.yellow);
       else
          cp.setBackground(Color.pink);
    }
}
