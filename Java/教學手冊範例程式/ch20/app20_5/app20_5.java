// app20_5, RGB color色階的應用
import java.awt.*;
import java.awt.event.*;
public class app20_5 extends Frame implements AdjustmentListener
{
   static app20_5 frm=new app20_5();
   static Scrollbar scr1=new Scrollbar(Scrollbar.VERTICAL);
   static Scrollbar scr2=new Scrollbar(Scrollbar.HORIZONTAL);
   static Scrollbar scr3=new Scrollbar(Scrollbar.VERTICAL);

   public static void main(String args[])
   {
      BorderLayout br=new BorderLayout(5,5);
      frm.setTitle("Display colors");
      frm.setSize(200,150);
      frm.add(scr1,br.WEST);           // scr1捲軸，用來控制紅色
      frm.add(scr2,br.SOUTH);          // scr2捲軸，用來控制綠色
      frm.add(scr3,br.EAST);           // scr3捲軸，用來控制藍色
      scr1.setValues(255,45,0,300);    // 設定scr1的相關數值
      scr2.setValues(255,45,0,300);    // 設定scr2的相關數值
      scr3.setValues(140,45,0,300);    // 設定scr3的相關數值
      scr1.addAdjustmentListener(frm);
      scr2.addAdjustmentListener(frm);
      scr3.addAdjustmentListener(frm);
      frm.setVisible(true);
   }
   public void adjustmentValueChanged(AdjustmentEvent e)
   {
      Graphics g=getGraphics();
      paint(g);
   }

   public void paint(Graphics g)
   {
      int red=scr1.getValue();         // 取得捲軸scr1的值
      int green=scr2.getValue();    // 取得捲軸scr2的值
      int blue=scr3.getValue();     // 取得捲軸scr3的值
      String str="Color("+red+","+green+","+blue+")";
      g.setColor(new Color(red,green,blue));    // 設定繪圖顏色
      g.fillRect(0,0,getWidth(),getHeight());
      g.setColor(Color.black);         // 設定繪圖顏色為黑色
      g.drawString(str,45,80);         // 於(45,80)處寫上字串
   }
}
