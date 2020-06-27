// app17_10, 核取方塊的應用(一)
import java.awt.*;
public class app17_10
{
   static Frame frm=new Frame("Checkbox class");
   static Checkbox ckb1=new Checkbox("Epson 5900L",true);
   static Checkbox ckb2=new Checkbox("HP LaserJet 4p",true);
   static Checkbox ckb3=new Checkbox("Other printer");

   public static void main(String args[])
   {
      frm.setSize(200,150);
      frm.setLayout(null);
      frm.setBackground(Color.yellow);
      ckb1.setBounds(20,40,140,20);    // 設定核取方塊的位置與大小
      ckb2.setBounds(20,60,140,20);
      ckb3.setBounds(20,80,140,20);
      frm.add(ckb1);             // 加入核取方塊到視窗中
      frm.add(ckb2);
      frm.add(ckb3);
      frm.setVisible(true);
   }
}
