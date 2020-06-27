// app19_1, 建立選擇表單
import java.awt.*;
public class app19_1 extends Frame
{
   static app19_1 frm=new app19_1();
   static List lst=new List();         // 建立選擇表單物件lst
   public static void main(String args[])
   {
      frm.setLayout(new FlowLayout(FlowLayout.CENTER,10,25));
      frm.setTitle("List class");
      for(int i=0;i<=9;i++)         // 利用for迴圈加入選項
         lst.add("List Item "+i);
      lst.select(2);             // 選取索引值為2的選項
      frm.setSize(200,150);
      frm.add(lst);
      frm.setBackground(Color.yellow);
      frm.setVisible(true);
      System.out.println("lst.getRows()= "+lst.getRows());
      System.out.println("lst.getItemCount()= "+lst.getItemCount());
   }
}

/* app19_1 OUTPUT-------
lst.getRows()= 4
lst.getItemCount()= 10
----------------------*/