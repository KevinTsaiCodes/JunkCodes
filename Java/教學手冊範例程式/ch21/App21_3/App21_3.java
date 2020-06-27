// App21_3, 簡單的applet程式
import java.awt.*;
import java.applet.Applet;

public class App21_3 extends Applet
{
   public void paint(Graphics g)
   {
       g.drawString(getParameter("str1"),30,40); // 取得HTML裡的str1字串
       g.drawString(getParameter("str2"),30,60); // 取得HTML裡的str2字串
   }
}