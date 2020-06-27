// app20_8, 填滿圓弧與圓形
import java.awt.*;
public class app20_8 extends Frame
{
   static app20_8 frm=new app20_8();

   public static void main(String args[])
   {
      frm.setTitle("Arc & Oval");
      frm.setSize(200,150);
      frm.setVisible(true);
   }

   public void paint(Graphics g)
   {
      g.setColor(Color.pink);       // 設定繪圖顏色為粉紅
      g.fillArc(20,50,70,70,30,300);   // 填滿圓弧
      g.setColor(Color.red);        // 設定繪圖顏色為紅色
      for(int x=70;x<=170;x=x+20)
         g.fillOval(x,80,10,10);    // 繪出六個小圓
   }
}
