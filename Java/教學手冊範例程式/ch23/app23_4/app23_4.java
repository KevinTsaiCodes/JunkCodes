// app23_4, JButton與JLabel的綜合應用
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

public class app23_4
{
   static JFrame frm=new JFrame("JButton & JLabel");
   static Container cp=frm.getContentPane();

   static ImageIcon pic[]=new ImageIcon[4];  // 建立ImageIcon陣列

   static ImageIcon left=new ImageIcon("c:\\Java\\left.gif");
   static ImageIcon right=new ImageIcon("c:\\Java\\right.gif");

   static JButton btn1=new JButton(" 前一張 ",left);
   static JButton btn2=new JButton(" 後一張 ",right);
   static JLabel lab=new JLabel();

   static int index=0;         // index變數，用來記錄哪一張影像正被顯示

   public static void main(String args[])
   {
       pic[0]=new ImageIcon("c:\\Java\\pic0.jpg"); // 載入影像
       pic[1]=new ImageIcon("c:\\Java\\pic1.jpg");
       pic[2]=new ImageIcon("c:\\Java\\pic2.jpg");
       pic[3]=new ImageIcon("c:\\Java\\pic3.jpg");

       cp.setLayout(new FlowLayout());
       btn2.setHorizontalTextPosition(JButton.LEFT); // 設定文字水平位置
       cp.add(btn1);
       cp.add(btn2);
       cp.add(lab);
       lab.setIcon(pic[0]);
       lab.setText("pic0.jpg");
       lab.setHorizontalTextPosition(JLabel.CENTER); // 設定文字水平位置
       lab.setVerticalTextPosition(JLabel.BOTTOM); // 設定文字垂直位置

       btn1.addActionListener(new ActLis());
       btn2.addActionListener(new ActLis());

       frm.setSize(400,350);
       frm.setVisible(true);
   }

   static class ActLis implements ActionListener
   {
      public void actionPerformed(ActionEvent e)
      {
         JButton btn=(JButton) e.getSource();   // 取得被按下的按鈕
         int num=pic.length;

         if(btn==btn1 && index>0)    // 若btn1被按下，且index>0
            index--;
         if(btn==btn2 && index<num-1) // 若btn2被按下，且index<num-1
            index++;

         lab.setText("pic"+ index%num +".jpg");  // 設定標題名稱
         lab.setIcon(pic[index%num]);
      }
   }
}
