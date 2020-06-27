// app19_2, 選擇表單的事件處理範例
import java.awt.*;
import java.awt.event.*;
public class app19_2 extends Frame implements ItemListener
{
   static app19_2 frm=new app19_2();
   static List lst=new List();            // 建立選擇表單物件lst
   public static void main(String args[])
   {
      frm.setLayout(new FlowLayout(FlowLayout.CENTER,10,25));
      frm.setTitle("Select a color");
      lst.add("yellow");               // 加入選項到lst內
      lst.add("orange");
      lst.add("pink");
      lst.add("cyan");
      lst.addItemListener(frm);        // 設定frm為lst的傾聽者
      frm.add(lst);
      frm.setSize(200,150);
      frm.setVisible(true);
   }
   public void itemStateChanged(ItemEvent e)    // 事件處理的程式碼
   {
      String color=lst.getSelectedItem();    // 取得被選取之選項名稱
      if(color=="yellow")              // 如果選項名稱為yellow
         frm.setBackground(Color.yellow);
      else if(color=="orange")            // 如果選項名稱為orange
         frm.setBackground(Color.orange);
      else if(color=="pink")              // 如果選項名稱為pink
         frm.setBackground(Color.pink);
      else if(color=="cyan")              // 如果選項名稱為cyan
         frm.setBackground(Color.cyan);
      frm.setTitle("you select "+color);     // 設定視窗frm的標題
   }
}
