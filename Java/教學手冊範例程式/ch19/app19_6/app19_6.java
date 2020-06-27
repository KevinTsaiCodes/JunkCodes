// app19_6, 功能表的事件處理範例
import java.awt.*;
import java.awt.event.*;
public class app19_6 extends Frame implements ActionListener
{
   static app19_6 frm=new app19_6();
   static Label lab=new Label("Tastes differ.",Label.CENTER);
   static MenuBar mb=new MenuBar();             // 建立MenuBar物件
   static Menu menu1=new Menu("Style");
   static Menu menu2=new Menu("Exit");
   static MenuItem mi1=new MenuItem("Plane");
   static MenuItem mi2=new MenuItem("Bold");
   static MenuItem mi3=new MenuItem("Italic");
   static MenuItem mi4=new MenuItem("Close window");
   public static void main(String args[])
   {
      frm.setTitle("Menu Demo");
      mb.add(menu1);
      mb.add(menu2);
      menu1.add(mi1);
      menu1.add(mi2);
      menu1.add(mi3);
      menu2.add(mi4);
      mi1.addActionListener(frm);   // 設定frm為mi1的事件傾聽者
      mi2.addActionListener(frm);   // 設定frm為mi2的事件傾聽者
      mi3.addActionListener(frm);   // 設定frm為mi3的事件傾聽者
      mi4.addActionListener(frm);   // 設定frm為mi4的事件傾聽者
      lab.setFont(new Font("Dialog",Font.PLAIN,24));
      frm.add(lab);
      frm.setSize(280,150);
      frm.setMenuBar(mb);
      frm.setVisible(true);
   }
   public void actionPerformed(ActionEvent e)  // 事件處理的程式碼
   {
      MenuItem mi=(MenuItem) e.getSource();     // 取得觸發事件的物件
      if(mi==mi1)                         // mi1觸發事件
         lab.setFont(new Font("Dialog",Font.PLAIN,24));
      else if(mi==mi2)                 // mi2觸發事件
         lab.setFont(new Font("Dialog",Font.BOLD,24));
      else if(mi==mi3)                 // mi3觸發事件
         lab.setFont(new Font("Dialog",Font.ITALIC,24));
      else if(mi==mi4)                 // mi4觸發事件
         frm.dispose();                // 關閉視窗
   }
}
