// app18_1, 簡單的事件處理範例(未加入事件處理)
import java.awt.*;
public class app18_1 extends Frame // 設定app18_1類別繼承自Frame類別
{
   static app18_1 frm=new app18_1(); // 建立app18_1類別的物件frm
   static Button btn=new Button("Click Me");

   public static void main(String args[])
   {
      frm.setLayout(new FlowLayout());
      frm.setTitle("Action Event");
      frm.setSize(200,150);
      frm.add(btn);
      frm.setVisible(true);
   }
}
