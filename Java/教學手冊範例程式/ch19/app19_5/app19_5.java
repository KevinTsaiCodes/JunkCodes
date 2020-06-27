// app19_5, 建立功能表
import java.awt.*;
public class app19_5
{
   static Frame frm=new Frame("Menu Demo");
   static MenuBar mb=new MenuBar();       // 建立MenuBar物件
   static Menu menu1=new Menu("Color");
   static Menu menu2=new Menu("Exit");
   static MenuItem mi1=new MenuItem("Yellow");
   static MenuItem mi2=new MenuItem("Orange");
   static MenuItem mi3=new MenuItem("Pink");
   static MenuItem mi4=new MenuItem("Close window");

   public static void main(String args[])
   {
      mb.add(menu1);                   // 將menu1加入mb中
      mb.add(menu2);                   // 將menu2加入mb中
      menu1.add(mi1);                  // 將mi1加入menu1中
      menu1.add(mi2);                  // 將mi2加入menu1中
      menu1.add(mi3);                  // 將mi3加入menu1中
      menu2.add(mi4);                  // 將mi4加入menu2中
      frm.setMenuBar(mb);              // 設定frm的功能表為mb
      frm.setSize(200,150);
      frm.setVisible(true);
   }
}
