// app18_3, 定義內部類別當成傾聽者
import java.awt.*;
import java.awt.event.*;
public class app18_3       // 主類別，注意此類別不需繼承Frame類別
{
   static Frame frm=new Frame("Action Event");
   static Button btn=new Button("Click Me");

   public static void main(String args[])
   {
      btn.addActionListener(new ActLis());
      frm.setLayout(new FlowLayout());
      frm.setSize(200,150);
      frm.add(btn);
      frm.setVisible(true);
   }
   // 定義內部類別ActLis，並實作ActionListener介面
   static class ActLis implements ActionListener
   {
      public void actionPerformed(ActionEvent e)   // 事件發生的處理動作
      {
         frm.setBackground(Color.yellow);
      }
   }
}
