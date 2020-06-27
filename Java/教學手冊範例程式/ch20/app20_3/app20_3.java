// app20_3, app20_2的修改版
import java.awt.*;
import java.awt.event.*;
public class app20_3 extends Frame implements ActionListener
{
   static app20_3 frm=new app20_3();
   static Button btn=new Button("Draw");
   boolean clicked=false;    // 宣告boolean變數，用來判別按鈕是否按下

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
      clicked=true;                 // 設定按鈕已被按下
      Graphics g=getGraphics();        // 取得視窗的繪圖區
      paint(g);
   }
   public void paint(Graphics g)
   {
      if(clicked)                   // 如果按鈕被按下
         g.drawRect(100,50,70,55);        // 繪出長方形
   }
}
