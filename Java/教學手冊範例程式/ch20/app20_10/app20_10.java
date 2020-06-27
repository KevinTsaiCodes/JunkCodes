// app20_10, 拖曳滑鼠繪圖
import java.awt.*;
import java.awt.event.*;
public class app20_10 extends Frame implements MouseMotionListener,MouseListener
{
   static app20_10 frm=new app20_10();
   int x1,x2,y1,y2;
   public static void main(String args[])
   {
      frm.setTitle("Free drawing");
      frm.setSize(200,150);
      frm.addMouseListener(frm);    // 設定MouseListener
      frm.addMouseMotionListener(frm); // 設定MouseMotionListener
      frm.setVisible(true);
   }
   public void mousePressed(MouseEvent e)
   {
      x1=e.getX();   // 取得滑鼠按下時的x座標 (繪圖起始點的x座標)
      y1=e.getY();   // 取得滑鼠按下時的y座標 (繪圖起始點的y座標)
   }
   public void mouseDragged(MouseEvent e)
   {
      Graphics g=getGraphics();
      x2=e.getX();               // 取得拖曳滑鼠時的x座標
      y2=e.getY();               // 取得拖曳滑鼠時的y座標
      g.drawLine(x1,y1,x2,y2);      // 繪出(x1,y1)到(x2,y2)的連線
      x1=x2;                  // 更新繪圖起始點的x座標
      y1=y2;                  // 更新繪圖起始點的y座標
   }
   public void mouseMoved(MouseEvent e){}
   public void mouseReleased(MouseEvent e){}
   public void mouseEntered(MouseEvent e){}
   public void mouseExited(MouseEvent e){}
   public void mouseClicked(MouseEvent e){}
}
