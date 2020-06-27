// app17_16, FlowLayout類別的使用
import java.awt.*;
public class app17_16
{
   static Frame frm=new Frame("Flow Layout");
   public static void main(String args[])
   {
      FlowLayout flow=new FlowLayout(FlowLayout.CENTER,5,10);
      frm.setLayout(flow);                // 設定版面配置為流動式
      frm.setSize(200,150);
      frm.setBackground(Color.yellow);
      frm.add(new TextField("This",8));            // 加入文字方塊
      frm.add(new TextField("is",6));           // 加入文字方塊
      frm.add(new TextField("Flow Layout",16));       // 加入文字方塊
      frm.setVisible(true);
   }
}
