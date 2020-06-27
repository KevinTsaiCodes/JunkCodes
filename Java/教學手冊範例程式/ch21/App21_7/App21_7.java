// App21_7, 按滑鼠右鍵繪出圓形
import java.awt.*;
import java.awt.event.*;
import java.applet.Applet;

public class App21_7 extends Applet implements MouseListener
{
   int x,y;                      // 滑鼠指標的x、y座標
   boolean clicked=false;

   public void init()
   {
        this.addMouseListener(this);   // 設定applet為自己本身的傾聽者
   }
   public void mouseClicked(MouseEvent e)
   {
      clicked=true;
      x=e.getX();                // 取得滑鼠按下之點的x座標
      y=e.getY();                // 取得滑鼠按下之點的y座標
      update(getGraphics());           // 清除繪圖區，然後重繪
   }
   public void paint(Graphics g)
   {
        if(clicked)
        {
           g.setColor(Color.pink);           // 設定顏色為粉紅色
           g.fillOval(x-20,y-20,40,40);      // 以按下的位置為圓心繪出圓形
        }
   }
      public void mouseEntered(MouseEvent e){}
      public void mouseExited(MouseEvent e){}
      public void mousePressed(MouseEvent e){}
      public void mouseReleased(MouseEvent e){}
}
