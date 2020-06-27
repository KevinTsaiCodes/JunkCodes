// app17_5, 建立視窗物件
import java.awt.*;
public class app17_5
{
   static Frame frm=new Frame("Frame class");

   public static void main(String args[])
   {
      frm.setSize(200,150);      // 設定視窗的大小為200*150
      frm.setLocation(100,50);      // 設定視窗位置為(100,50)
      frm.setVisible(true);      // 設定視窗為可見

      System.out.println("state="+frm.getState());
      System.out.println("title="+frm.getTitle());
      System.out.println("visible="+frm.isVisible());
   }
}

/* app17_5 OUTPUT---
state=0
title=Frame class
visible=true
------------------*/