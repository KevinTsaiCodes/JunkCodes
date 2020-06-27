// app18_4, 加入可關閉視窗的按鈕
import java.awt.*;
import java.awt.event.*;
public class app18_4 extends Frame implements ActionListener
{
   static app18_4 frm=new app18_4();
   static Button btn1=new Button("Yellow");
   static Button btn2=new Button("Green");
   static Button btn3=new Button("Exit");

   public static void main(String args[])
   {
      btn1.addActionListener(frm);     // 把事件傾聽者frm向btn1註冊
      btn2.addActionListener(frm);     // 把事件傾聽者frm向btn2註冊
      btn3.addActionListener(frm);     // 把事件傾聽者frm向btn3註冊

      frm.setTitle("Action Event");
      frm.setLayout(new FlowLayout(FlowLayout.CENTER));
      frm.setSize(200,150);
      frm.add(btn1);
      frm.add(btn2);
      frm.add(btn3);
      frm.setVisible(true);
   }

   public void actionPerformed(ActionEvent e)
   {
      Button btn=(Button) e.getSource();     // 取得事件來源的物件
      if(btn==btn1)                       // 如果是按下btn1按鈕
         frm.setBackground(Color.yellow);
      else if(btn==btn2)               // 如果是按下btn2按鈕
         frm.setBackground(Color.green);
      else                                // 如果是按下btn3按鈕
         System.exit(0);
   }
}
