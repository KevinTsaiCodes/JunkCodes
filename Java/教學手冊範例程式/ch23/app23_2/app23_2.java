// app23_2, JInternalFrame類別的練習
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

public class app23_2
{
   static JFrame frm=new JFrame("JInternalFrame");
   static JButton btn=new JButton("New Frame"); // 建立JButton物件

   static Container cp=frm.getContentPane();   // 取得內容層
   static JDesktopPane jdp=new JDesktopPane(); // 建立桌面層物件

   public static void main(String args[])
   {
       cp.setLayout(new BorderLayout());
       cp.add(btn,BorderLayout.SOUTH);
       cp.add(jdp);        // 將桌面層加到內容層中

       btn.addActionListener(new ActLis());
       frm.setSize(400,300);
       frm.setVisible(true);
   }

   static class ActLis implements ActionListener
   {
      static int count=1;     // 宣告count變數，用來記錄子視窗的總數
      public void actionPerformed(ActionEvent e)
      {
         JInternalFrame jif;  // 建立子視窗物件jif
         jif=new JInternalFrame("Frame "+(count++),true,true,true,true);
         Container icp=jif.getContentPane();  // 取得jif的內容層
         JButton ibtn=new JButton("JInternalFrame Button");
         icp.add(ibtn,BorderLayout.SOUTH);       // 將ibtn按鈕加入icp中
         jdp.add(jif);        // 將子視窗物件jif加到桌面層中
         jif.setSize(200,150);
         jif.setVisible(true);
      }
   }
}
