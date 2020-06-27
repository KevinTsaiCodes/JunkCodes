// app20_9, 繪製多邊形
import java.awt.*;
class app20_9 extends Frame
{
   static app20_9 frm=new app20_9();

   public static void main(String args[])
   {
      frm.setTitle("Polygon");
      frm.setSize(200,150);
      frm.setVisible(true);
   }

   public void paint(Graphics g)
   {
      int x[]={44,65,97,139,58}; // 儲存所有頂點x座標的一維陣列
      int y[]={34,55,40,109,127};   // 儲存所有頂點y座標的一維陣列
      g.setColor(Color.pink);       // 繪圖顏色設為粉紅色
      g.fillPolygon(x,y,5);      // 繪出多邊形，並填滿粉紅色
      g.setColor(Color.black);   // 繪圖顏色設為黑色
      g.drawPolygon(x,y,5);         // 用黑色繪出多邊形
   }
}
