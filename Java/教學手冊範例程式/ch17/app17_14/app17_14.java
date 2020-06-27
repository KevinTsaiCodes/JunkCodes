// app17_14, BorderLayout類別的使用
import java.awt.*;
public class app17_14
{
   static Frame frm=new Frame("Border Layout");
   public static void main(String args[])
   {
      BorderLayout border=new BorderLayout(2,5);       // 建構元
      frm.setLayout(border);   // 將版面配置設定為BorderLayout
      frm.setSize(200,150);
      frm.add(new Button("East"),border.EAST);
      frm.add(new Button("West"),border.WEST);
      frm.add(new Button("South"),border.SOUTH);
      frm.add(new Button("North"),border.NORTH);
      frm.add(new Button("Center"),border.CENTER);
      frm.setVisible(true);
   }
}
