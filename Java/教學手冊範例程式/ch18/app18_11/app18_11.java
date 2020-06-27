// app18_11, 以MouseAdapter類別事件處理MouseEvent事件
import java.awt.*;
import java.awt.event.*;
public class app18_11 extends Frame
{
   static app18_11 frm=new app18_11();
   static Button btn=new Button(" Click Me!! ");
   static TextArea txa=new TextArea("",2,5,TextArea.SCROLLBARS_NONE);
   static MouseLis mlis=new MouseLis();     // 建立MouseLis類別的物件
   public static void main(String args[])
   {
      BorderLayout br=new BorderLayout(2,5);
      frm.setSize(200,150);
      frm.setTitle("Mouse Event");
      frm.setLayout(br);
      btn.addMouseListener(mlis);      // 以mlis物件做為btn的傾聽者
      txa.setEditable(false);
      frm.add(btn,br.WEST);
      frm.add(txa,br.CENTER);
      frm.setVisible(true);
   }
   // 定義 MouseLis為static類別，並繼承自MouseAdapter類別
   static class MouseLis extends MouseAdapter
   {
      public void mouseClicked(MouseEvent e)
      {
         int x=e.getX();            // 取得x座標
         int y=e.getY();            // 取得y座標
         txa.append("clicked at ("+x+","+y+")\n");
      }
   }
}
