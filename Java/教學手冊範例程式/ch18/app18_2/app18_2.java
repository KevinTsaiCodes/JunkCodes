// app18_2, 簡單的事件處理範例(已加入事件處理)
import java.awt.*;
import java.awt.event.*;
public class app18_2 extends Frame implements ActionListener
{
   static app18_2 frm=new app18_2();
   static Button btn=new Button("Click Me");

   public static void main(String args[])
   {
      btn.addActionListener(frm);            // 把frm向btn註冊
      frm.setLayout(new FlowLayout());
      frm.setTitle("Action Event");
      frm.setSize(200,150);
      frm.add(btn);
      frm.setVisible(true);
   }

   public void actionPerformed(ActionEvent e) // 事件發生的處理動作
   {
      frm.setBackground(Color.yellow);
   }
}
