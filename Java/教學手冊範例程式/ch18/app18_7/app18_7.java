// app18_7, 以KeyListener介面處理KeyEvent事件
import java.awt.*;
import java.awt.event.*;
public class app18_7 extends Frame implements KeyListener
{
   static app18_7 frm=new app18_7();
   static TextField txf=new TextField(18);
   static TextArea txa=new TextArea("",4,19,TextArea.SCROLLBARS_NONE);

   public static void main(String args[])
   {
      frm.setSize(200,150);
      frm.setTitle("Key Event");
      frm.setLayout(new FlowLayout(FlowLayout.CENTER));
      txf.addKeyListener(frm);      // 將frm設為txf的傾聽者
      txa.setEditable(false);       // 將txa設為不可編輯
      frm.add(txf);
      frm.add(txa);
      frm.setVisible(true);
   }
   // 當txf物件觸發KeyEvent事件時，依事件種類執行下列的程式碼
   public void keyPressed(KeyEvent e)  // 當按鍵按下時
   {
      txa.setText("");           // 清空txa裡的文字
      txa.append("keyPressed() called\n");
   }
   public void keyReleased(KeyEvent e) // 當按鍵放開時
   {
      txa.append("keyReleased() called\n");
   }
   public void keyTyped(KeyEvent e) // 鍵入文字時
   {
      txa.append("keyTyped() called\n");
   }
}
