// App20_1, 虏虫applet祘Α
import java.awt.*;
import java.applet.Applet;

public class App21_5 extends Applet
{
   public void init()               // init() ㄧ计
   {
       System.out.println("init() method called!");
   }
   public void start()              // start() ㄧ计
   {
       System.out.println("start() method called!");
   }
   public void stop()               // stop() ㄧ计
   {
       System.out.println("stop() method called!");
   }
   public void destroy()            // destroy() ㄧ计
   {
       System.out.println("destroy() method called!");
   }
   public void paint(Graphics g)
   {
       g.drawString("Love makes the world go round.",30,50);
       System.out.println("paint() called!");
   }
}
