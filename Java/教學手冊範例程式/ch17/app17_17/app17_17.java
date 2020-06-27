// app17_17, GridLayout類別的使用
import java.awt.*;
public class app17_17
{
   static Frame frm=new Frame("Grid Layout");
   public static void main(String args[])
   {
      GridLayout grid=new GridLayout(3,5);      // 3列5行的配置
      frm.setLayout(grid);
      frm.setSize(200,150);
      for(int i=1;i<=15;i++)
         frm.add(new Button(Integer.toString(i)));    // 加入按鈕
      frm.setVisible(true);
   }
}
