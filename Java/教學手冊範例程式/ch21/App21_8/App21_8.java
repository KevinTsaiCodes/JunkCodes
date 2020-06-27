// App21_8, 在applet裡載入圖檔
import java.awt.*;
import java.awt.event.*;
import java.applet.Applet;

public class App21_8 extends Applet
{
   Image img;                 // 宣告Image類別型態的變數img

   public void init()
   {
        img=getImage(getCodeBase(),"flower.jpg");        // 載入圖檔
   }

   public void paint(Graphics g)
   {
      g.drawImage(img,20,20,this);  // 將img畫在applet上
   }
}
