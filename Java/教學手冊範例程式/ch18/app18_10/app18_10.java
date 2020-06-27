// app18_10, 以MouseMotionListener介面處理MouseEvent事件
import java.awt.*;
import java.awt.event.*;
public class app18_10 extends Frame implements MouseMotionListener
{
   static app18_10 frm=new app18_10();
   static Label labx=new Label();
   static Label laby=new Label();
   static Label lab=new Label();
   public static void main(String agrs[])
   {
      frm.setLayout(null);
      frm.addMouseMotionListener(frm); // 設定frm為事件的傾聽者
      labx.setBounds(40,40,40,20);
      laby.setBounds(100,40,40,20);
      lab.setBounds(40,80,100,40);
      frm.setSize(200,150);
      frm.setTitle("Mouse Event");
      frm.add(labx);
      frm.add(laby);
      frm.add(lab);
      frm.setVisible(true);
   }
   public void mouseMoved(MouseEvent e)   // 當滑鼠移動時
   {
      labx.setText("x="+e.getX());     // 顯示x座標
      laby.setText("y="+e.getY());     // 顯示y座標
      lab.setText("Mouse Moved!!");    // 顯示"Mouse Moved!!"字串
   }
   public void mouseDragged(MouseEvent e) // 當滑鼠拖曳時
   {
      labx.setText("x="+e.getX());     // 顯示x座標
      laby.setText("y="+e.getY());     // 顯示y座標
      lab.setText("Mouse Dragged!!");  // 顯示"Mouse Dragged!!"字串
   }
}
