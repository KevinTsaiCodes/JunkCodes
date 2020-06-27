// app18_9, 以MouseListener介面處理MouseEvent事件
import java.awt.*;
import java.awt.event.*;
public class app18_9 extends Frame implements MouseListener
{
   static app18_9 frm=new app18_9();
   static Button btn=new Button(" Click Me!! ");
   static TextArea txa=new TextArea("",2,5,TextArea.SCROLLBARS_NONE);

   public static void main(String args[])
   {
      BorderLayout br=new BorderLayout(2,5);
      frm.setSize(200,150);
      frm.setTitle("Mouse Event");
      frm.setLayout(br);
      btn.addMouseListener(frm);          // 設定frm為btn的傾聽者
      txa.setEditable(false);
      frm.add(btn,br.WEST);
      frm.add(txa,br.CENTER);
      frm.setVisible(true);
   }

   public void mouseEntered(MouseEvent e) // 滑鼠的指標進入btn上方
   {
      txa.setText("Mouse entered\n");
   }
   public void mouseClicked(MouseEvent e) // 按下並放開滑鼠按鈕
   {
      txa.append("Mouse clicked\n");
   }
   public void mouseExited(MouseEvent e)   // 滑鼠的指標移開btn上方
   {
      txa.append("Mouse exited\n");
   }
   public void mousePressed(MouseEvent e) // 按下滑鼠按鈕
   {
      txa.append("Mouse pressed\n");
   }
   public void mouseReleased(MouseEvent e) // 放開滑鼠按鈕
   {
      txa.append("Mouse released\n");
   }
}
