// app17_12, TextField的應用
import java.awt.*;
public class app17_12
{
   static Frame frm=new Frame("TextField class");
   static TextField txf1=new TextField("TextField Demo");
   static TextField txf2=new TextField("Editable");
   static TextField txf3=new TextField("password");

   public static void main(String args[])
   {
      frm.setSize(200,150);
      frm.setLayout(null);
      frm.setBackground(Color.yellow);
      txf1.setBounds(20, 40,120,20);
      txf2.setBounds(20, 70,120,20);
      txf3.setBounds(20,100,120,20);
      txf1.setEditable(false);         // 設定txf1為不可編輯
      txf3.setEchoChar('*');           // 設定txf3的回應字元為'*'
      frm.add(txf1);
      frm.add(txf2);
      frm.add(txf3);
      System.out.println(txf1.getText());
      System.out.println(txf2.getText());
      System.out.println(txf3.getText());
      frm.setVisible(true);
   }
}

/* app17_12 OUTPUT---
TextField Demo
Editable
password
-------------------*/