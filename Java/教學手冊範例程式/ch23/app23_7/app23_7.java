// app23_7, JList的練習（二）
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import java.util.Vector;         // 載入util類別庫裡的Vector類別

public class app23_7
{
   static JFrame frm=new JFrame("JList demo");
   static Container cp=frm.getContentPane();
   static JList<String> lst1=new JList<String>();  // 建立lst1物件
   static JList<String> lst2=new JList<String>();  // 建立lst2物件
   static String s[]={"RED","GREEN","BLUE","PINK","YELLOW","CYAN","GRAY"};
   static Vector<String> v=new Vector<String>(); // 建立Vector類別的物件v

   public static void main(String args[])
   {
      cp.setLayout(new GridLayout(1,2));
      cp.add(new JScrollPane(lst1));   // 將JScrollPane加入cp中
      cp.add(new JScrollPane(lst2));   // 將JScrollPane加入cp中
      lst1.setListData(s);                // 設定lst1物件的選單
      lst1.addMouseListener(new MouseLis());    // 設定lst1物件的傾聽者
      frm.setSize(200,155);
      frm.setVisible(true);
   }
   static class MouseLis extends MouseAdapter
   {
      public void mouseClicked(MouseEvent e)
      {
         if(e.getSource()==lst1)          // 若是lst1物件被按下
            if(e.getClickCount()==2)         // 如果連續被按了兩下
            {
               int index=lst1.getSelectedIndex();
               String str=s[index];
               v.add(str);                   // 將字串str加入向量v
               lst2.setListData(v);    // 設定向量v為lst2物件的選單
            }
      }
   }
}
