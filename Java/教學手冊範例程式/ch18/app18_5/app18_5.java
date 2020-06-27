// app18_5, ItemEvent類別的使用範例
import java.awt.*;
import java.awt.event.*;
public class app18_5 extends Frame implements ItemListener
{
   static app18_5 frm=new app18_5();
   static Checkbox ckb1=new Checkbox("Epson 5900L");
   static Checkbox ckb2=new Checkbox("HP LaserJet 4p");
   static Label lab=new Label(" Select a printer ");

   public static void main(String args[])
   {
      CheckboxGroup grp=new CheckboxGroup();
      frm.setSize(200,150);
      frm.setTitle("Item Event");
      frm.setLayout(new FlowLayout(FlowLayout.LEFT));
      ckb1.setCheckboxGroup(grp);      // 將ckb1設為單選
      ckb2.setCheckboxGroup(grp);      // 將ckb2設為單選
      lab.setBackground(Color.orange);
      ckb1.addItemListener(frm);       // 讓frm當成ckb1的傾聽者
      ckb2.addItemListener(frm);       // 讓frm當成ckb2的傾聽者
      frm.add(lab);
      frm.add(ckb1);
      frm.add(ckb2);
      frm.setVisible(true);
   }
   // ItemEvent事件發生時的處理動作
   public void itemStateChanged(ItemEvent e)
   {
      if(ckb1.getState()==true)        // 如果是ckb1被選擇
         lab.setText(" Epson 5900L");
      else if(ckb2.getState()==true)   // 如果是ckb2被選擇
         lab.setText(" HP LaserJet 4p");
   }
}
