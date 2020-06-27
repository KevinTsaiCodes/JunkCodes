// app18_12, WindowAdapter類別來處理視窗事件
import java.awt.*;
import java.awt.event.*;
public class app18_12 extends Frame
{
   static app18_12 frm=new app18_12();
   static WinLis wlis=new WinLis();    // 產生WinLis類別的物件wlis

   public static void main(String args[])
   {
      frm.setSize(200,150);
      frm.setTitle("Window Event");
      frm.addWindowListener(wlis);     // 設定wlis為frm的事件傾聽者
      frm.setVisible(true);
   }
   // 定義 WinLis為static，並繼承自WindowAdapter類別
   static class WinLis extends WindowAdapter
   {
      public void windowClosing(WindowEvent e)  // 按下視窗關閉鈕
      {
         System.out.println("windowClosing() called");
         System.out.println("Closing window...");
         frm.dispose();                      // 關閉視窗並釋放資源
         System.out.println("window closed...");
      }
      public void windowClosed(WindowEvent e)   // 關閉視窗
      {
         System.out.println("windowClosed() called");
      }
      public void windowDeactivated(WindowEvent e)    // 變成非作用中視窗
      {
         System.out.println("windowDeactivated() called");
      }
      public void windowActivated(WindowEvent e)   // 變成作用中視窗
      {
         System.out.println("windowActivated() called");
      }
      public void windowDeiconified(WindowEvent e)    // 視窗還原
      {
         System.out.println("windowDeiconified() called");
      }
      public void windowIconified(WindowEvent e)      // 視窗最小化
      {
         System.out.println("windowIconified() called");
      }
      public void windowOpened(WindowEvent e)      // 開啟視窗
      {
         System.out.println("windowOpened() called");
      }
   }
}
