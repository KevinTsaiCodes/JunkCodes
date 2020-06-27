// app20_11, 利用滑鼠移動圓形
import java.awt.*;
import java.awt.event.*;
public class app20_11 extends Frame implements MouseMotionListener,MouseListener
{
   static app20_11 frm=new app20_11();
   int x=70,y=60,posX=70,posY=60,dx,dy;
   public static void main(String args[])
   {
      frm.setTitle("Dragging a circle");
      frm.setSize(200,150);
      frm.addMouseListener(frm);
      frm.addMouseMotionListener(frm);
      frm.setVisible(true);
   }
   public void mousePressed(MouseEvent e)
   {
      dx=e.getX()-posX;       // 取得滑鼠按下之點與基準點x方向的距離
      dy=e.getY()-posY;       // 取得滑鼠按下之點與基準點y方向的距離
   }
   public void mouseDragged(MouseEvent e)
   {
      x=e.getX()-dx;          // 取得拖曳時的基準點x座標
      y=e.getY()-dy;          // 取得拖曳時的基準點y座標
      if(dx>0 && dx<50 && dy>0 && dy<50) // 如果指標落在正方形區域內
      {
         Graphics g=getGraphics();
         update(g);           // 清空畫面為背景顏色，再呼叫paint()
      }
   }
   public void paint(Graphics g)
   {
      g.setColor(Color.pink);  // 設定繪圖顏色為粉紅
      g.fillOval(x,y,50,50);   // 以基準點為圖形的左上角繪出圓形
      posX=x;              // 更新基準點的x座標
      posY=y;              // 更新基準點的y座標
   }
   public void mouseMoved(MouseEvent e){}
   public void mouseReleased(MouseEvent e){}
   public void mouseEntered(MouseEvent e){}
   public void mouseExited(MouseEvent e){}
   public void mouseClicked(MouseEvent e){}
}
