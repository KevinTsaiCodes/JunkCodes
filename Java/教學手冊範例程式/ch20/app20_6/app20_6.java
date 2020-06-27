// app20_6, brighter()與darker()的使用
import java.awt.*;
import java.awt.event.*;
class app20_6 extends Frame implements ActionListener
{
   static app20_6 frm=new app20_6();
   static Button btn1=new Button("Brighter");   // Brighter按鈕
   static Button btn2=new Button("Darker");     // Darker按鈕
   Color co=new Color(255,255,255);       // 設定顏色的初值

   public static void main(String args[])
   {
      frm.setTitle("Brighter & Darker");
      frm.setLayout(new FlowLayout());
      frm.setSize(200,150);
      frm.add(btn1);
      frm.add(btn2);
      btn1.addActionListener(frm);
      btn2.addActionListener(frm);
      frm.setVisible(true);
   }
   public void actionPerformed(ActionEvent e)
   {
      Button btn=(Button) e.getSource();        // 取得被按下的按鈕
      if(btn==btn1)
         co=co.brighter();   // 如果是按下brighter，則將顏色變亮一點
      else if(btn==btn2)
         co=co.darker();     // 如果是按下Darker，則將顏色變暗一點
      Graphics g=getGraphics();
      paint(g);
   }
   public void paint(Graphics g)
   {
      g.setColor(co);                     // 將繪圖顏色設為co
      g.fillRect(0,0,getWidth(),getHeight());      // 填滿顏色
   }
}
