// app20_4, 簡單的繪圖
import java.awt.*;
import java.awt.event.*;
public class app20_4 extends Frame  // 設定app20_4繼承自Frame類別
{
   static app20_4 frm=new app20_4();         // 建立視窗物件frm

   public static void main(String args[])
   {
      frm.setTitle("Drawing example");
      frm.setSize(200,150);
      frm.setVisible(true);
   }
   public void paint(Graphics g)
   {
      g.setFont(new Font("Arial",Font.ITALIC,18)); // 設定使用的字型
      g.drawString("Live and learn.",45,88);    // 繪出字串
      g.setColor(Color.red);                 // 設定繪圖顏色為紅色
      g.drawRect(30,65,145,30);           // 繪出長方形
   }
}
