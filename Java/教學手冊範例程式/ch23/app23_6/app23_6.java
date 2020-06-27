// app23_6, JList的練習（一）
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*;

public class app23_6
{
   static JFrame frm=new JFrame("JList demo");
   static Container cp=frm.getContentPane();
   static JButton btn=new JButton("Get Colors");
   static JList<String> lst=new JList<String>();       // 建立JList物件

   public static void main(String args[])
   {
      cp.setLayout(new BorderLayout());
      cp.add(btn,BorderLayout.SOUTH);
      cp.add(new JScrollPane(lst));    // 將lst加入JScrollPane中
      btn.addActionListener(new ActLis());      // 設定btn的傾聽者
      lst.addListSelectionListener(new LSLis()); // 設定lst的傾聽者
      frm.setSize(200,155);
      frm.setVisible(true);
   }
   static class ActLis implements ActionListener
   {
      public void actionPerformed(ActionEvent e)
      {
         String s[]={"RED","GREEN","BLUE","PINK","YELLOW","CYAN","GRAY"};
         lst.setListData(s); // 將陣列s的內容加入lst中，做為lst的選項
      }
   }
   static class LSLis implements ListSelectionListener
   {
      public void valueChanged(ListSelectionEvent e)
      {
         int color=lst.getSelectedIndex();   // 取得被選取選項的註標
         switch(color)
         {
            case 0: lst.setBackground(Color.RED);     break;
            case 1: lst.setBackground(Color.GREEN);   break;
            case 2: lst.setBackground(Color.BLUE); break;
            case 3: lst.setBackground(Color.PINK); break;
            case 4: lst.setBackground(Color.YELLOW);  break;
            case 5: lst.setBackground(Color.CYAN); break;
            case 6: lst.setBackground(Color.GRAY); break;
         }
      }
   }
}
