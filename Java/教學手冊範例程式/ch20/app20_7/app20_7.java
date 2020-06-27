// app20_7, 列出系統所提供的字型
import java.awt.*;
import java.awt.event.*;
class app20_7 extends Frame implements ItemListener
{
   static app20_7 frm=new app20_7();
   static List lst=new List();      // 建立一個List物件
   String str="Arial";        // 設定字串的初值

   public static void main(String args[])
   {
      BorderLayout br=new BorderLayout(5,5);
      frm.add(lst,br.NORTH);
      frm.setBackground(Color.yellow);
      frm.setTitle("Font List");
      lst.addItemListener(frm);
      frm.setSize(300,150);
      GraphicsEnvironment ge;
      ge=GraphicsEnvironment.getLocalGraphicsEnvironment();
      String fnt[]=ge.getAvailableFontFamilyNames();
      for(int i=2;i<fnt.length-2;i++)
         lst.add(fnt[i]);
      frm.setVisible(true);
   }
   public void itemStateChanged(ItemEvent e)
   {
      str=lst.getSelectedItem(); // 取得選擇表單裡被選取的項目
      Graphics g=getGraphics();
      update(g);
   }
   public void paint(Graphics g)
   {
      g.setFont(new Font(str,Font.PLAIN,20));   // 設定字型
      g.setColor(Color.black);               // 設定顏色
      g.drawString(str,20,110);     // 用指定的顏色與字型寫上字串
   }
}
