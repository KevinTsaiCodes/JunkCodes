// app23_5, 核取方塊與選項方塊的應用
import java.awt.*;
import javax.swing.*;

public class app23_5
{
   static JFrame frm=new JFrame("Checkbox class");
   static Container cp=frm.getContentPane();
   static JRadioButton rb1=new JRadioButton("數位攝影機");
   static JRadioButton rb2=new JRadioButton("數位相機");

   static JCheckBox ckb1=new JCheckBox("Sony",true);
   static JCheckBox ckb2=new JCheckBox("Nikon",true);
   static JCheckBox ckb3=new JCheckBox("Others");

   public static void main(String args[])
   {
      rb1.setBounds(10,20,90,20);
      rb2.setBounds(107,20,78,20);
      ckb1.setBounds(20,60,140,20);
      ckb2.setBounds(20,80,140,20);
      ckb3.setBounds(20,100,140,20);

      ButtonGroup bgroup=new ButtonGroup();  // 建立ButtonGroup物件
      bgroup.add(rb1);                 // 將rb1設定為單選
      bgroup.add(rb2);                 // 將rb2設定為單選
      rb1.setSelected(true);              // 設定rb1被選擇

      cp.add(rb1);
      cp.add(rb2);
      cp.add(ckb1);
      cp.add(ckb2);
      cp.add(ckb3);
      cp.setLayout(null);
      cp.setBackground(Color.white);
      frm.setSize(200,170);
      frm.setVisible(true);
   }
}
