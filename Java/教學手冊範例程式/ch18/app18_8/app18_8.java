// app18_8, 以KeyAdapter類別事件處理KeyEvent事件
import java.awt.*;
import java.awt.event.*;
public class app18_8 extends Frame   // 不需實作KeyListener介面
{
   static app18_8 frm=new app18_8();
   static TextField txf=new TextField(18);
   static TextArea txa=new TextArea("",4,19,TextArea.SCROLLBARS_NONE);
   public static void main(String args[])
   {
      frm.setSize(200,150);
      frm.setTitle("Key Event");
      frm.setLayout(new FlowLayout(FlowLayout.CENTER));
      txf.addKeyListener(new KeyLis());
      txa.setEditable(false);
      frm.add(txf);
      frm.add(txa);
      frm.setVisible(true);
   }
   // 定義 KeyLis為static類別，並繼承自KeyAdapter類別
   static class KeyLis extends KeyAdapter
   {
      public void keyPressed(KeyEvent e)
      {
         txa.setText("");     // 清空txa裡的文字
         if(e.isActionKey())  // 如果是Action key
            txa.append("Action key is pressed\n");
         else                 // 如果不是Action key，則印出字元
            txa.append(e.getKeyChar()+" is pressed\n");
      }
   }
}