// app19_4, 下拉選單的事件處理範例
import java.awt.*;
import java.awt.event.*;
class app19_4 extends Frame implements ItemListener
{
   static app19_4 frm=new app19_4();
   static Choice chc=new Choice();   // 建立下拉表單物件chc
   public static void main(String args[])
   {
      BorderLayout br=new BorderLayout();
      frm.setLayout(br);
      frm.setTitle("Select a color");
      chc.add("yellow");         // 加入選項到下拉表單物件chc裡
      chc.add("orange");
      chc.add("pink");
      chc.add("cyan");
      chc.addItemListener(frm);     // 設定frm為chc的事件傾聽者
      frm.setSize(200,150);
      frm.add(chc,br.NORTH);
      frm.setVisible(true);
   }
   public void itemStateChanged(ItemEvent e)
   {
      String color=chc.getSelectedItem();
      if(color=="yellow")        // 選擇yellow選項
         frm.setBackground(Color.yellow);
      else if(color=="orange")      // 選擇orange選項
         frm.setBackground(Color.orange);
      else if(color=="pink")        // 選擇pink選項
         frm.setBackground(Color.pink);
      else if(color=="cyan")        // 選擇cyan選項
         frm.setBackground(Color.cyan);
      frm.setTitle("you select "+color);
   }
}
