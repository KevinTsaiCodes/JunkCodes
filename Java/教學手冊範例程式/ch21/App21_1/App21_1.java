// App21_1, 簡單的applet程式
import java.awt.*;
import java.applet.Applet;          // 載入Applet類別

public class App21_1 extends Applet    // App21_1衍生自Applet類別
{
   public void paint(Graphics g)
   {
       g.setColor(Color.blue);      // 設定繪圖顏色為藍色
       g.fillOval(30,30,50,50);     // 繪出圓形並填滿藍色
       g.setColor(Color.orange);    // 設定繪圖顏色為橘色
       g.fillOval(60,40,90,90);     // 繪出圓形並填滿橘色
   }
}
