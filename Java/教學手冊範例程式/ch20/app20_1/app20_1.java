// app20_1, 簡單的繪圖
import java.awt.*;
import java.awt.event.*;
public class app20_1 extends Frame implements ActionListener
{
   static app20_1 frm=new app20_1();
   static Button btn=new Button("Draw");

   public static void main(String args[])
   {
      BorderLayout br=new BorderLayout();
      frm.setTitle("Drawing example");
      frm.setLayout(br);
      frm.setSize(200,150);
      frm.add(btn,br.SOUTH);
      btn.addActionListener(frm);
      frm.setVisible(true);
   }
   public void actionPerformed(ActionEvent e)
   {
      Graphics g=getGraphics();        // 取得視窗的繪圖區
      g.drawRect(100,50,70,55);        // 繪出長方形
   }
}
