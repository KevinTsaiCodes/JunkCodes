// app19_7, 捲軸的實例應用
import java.awt.*;
import java.awt.event.*;
public class app19_7 extends Frame implements AdjustmentListener
{
   static app19_7 frm=new app19_7();
   static Scrollbar scr=new Scrollbar();  // 建立垂直捲軸scr
   static Label lab1=new Label("Silence is golden.",Label.CENTER);
   static Label lab2=new Label("size=20",Label.CENTER);

   public static void main(String args[])
   {
      BorderLayout br=new BorderLayout(5,5);
      frm.setTitle("Scrollbar Demo");
      frm.setSize(300,150);
      scr.addAdjustmentListener(frm);  // 以frm當成scr的傾聽者
      scr.setValues(20,4,12,40);          // 設定scr的相關數值
      frm.add(scr,br.EAST);
      frm.add(lab1,br.CENTER);
      frm.add(lab2,br.SOUTH);
      lab1.setFont(new Font("Dialog",Font.PLAIN,20));
      lab2.setBackground(Color.orange);
      frm.setVisible(true);
   }
   public void adjustmentValueChanged(AdjustmentEvent e)
   {
      int size=scr.getValue();      // 取得scr的數值
      lab1.setFont(new Font("Dialog",Font.PLAIN,size));   // 設定字型樣式
      lab2.setText("size="+size);      // 顯示字體大小
   }
}
