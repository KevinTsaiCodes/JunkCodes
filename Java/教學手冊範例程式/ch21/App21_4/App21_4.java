// App21_4, 簡單的applet程式
import java.awt.*;
import java.applet.Applet;

public class App21_4 extends Applet
{
   public void paint(Graphics g)
   {
       g.drawString("Experience is the best teacher.",30,50);
       System.out.println("printed from Java console!");
   }
}
