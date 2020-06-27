// app23_3, JButton影像圖示的變化
import java.awt.*;
import javax.swing.*;

public class app23_3
{
   static JFrame frm=new JFrame("JButton 測試");
   static Container cp=frm.getContentPane();

   static ImageIcon general=new ImageIcon("c:\\Java\\img1.gif");
   static ImageIcon rollover=new ImageIcon("c:\\Java\\img2.gif");
   static ImageIcon pressed=new ImageIcon("c:\\Java\\img3.gif");
   static JButton btn=new JButton("JButton");  // 建立JButton物件

   public static void main(String args[])
   {
       cp.setLayout(new FlowLayout());
       cp.add(btn);  // 將按鈕加入內容層中

       btn.setRolloverEnabled(true);   // 設定滑鼠指標與按鈕有互動效果
       btn.setIcon(general);     // 設定在一般情況下，按鈕的圖示
       btn.setRolloverIcon(rollover);  // 設定指標在按鈕上方時的圖示
       btn.setPressedIcon(pressed); // 設定滑鼠按鍵按下時的圖示

       frm.setSize(200,130);
       frm.setVisible(true);
   }
}
