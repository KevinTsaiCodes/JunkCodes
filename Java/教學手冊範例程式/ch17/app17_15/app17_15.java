// app17_15, CardLayout類別的用法
import java.awt.*;
public class app17_15
{
   static Frame frm=new Frame("Card Layout");
   public static void main(String args[])
   {
      CardLayout card=new CardLayout(5,10);      // 使用多層版面配置
      frm.setLayout(card);
      frm.setSize(200,150);
      frm.add(new Button("Button 1"),"c1");
      frm.add(new Button("Button 2"),"c2");
      frm.add(new Button("Button 3"),"c3");
      card.show(frm,"c2");
      frm.setVisible(true);
   }
}
