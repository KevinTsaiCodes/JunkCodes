// app19_9, FileDialog類別的使用
import java.awt.*;
import java.awt.event.*;
import java.io.*;
public class app19_9 extends Frame implements ActionListener
{
   static app19_9 frm=new app19_9();
   static FileDialog fdlg=new FileDialog(frm,"Open"); // 建立fdlg物件
   static Button btn=new Button("Open");
   static TextArea txa=new TextArea();

   public static void main(String args[])
   {
      BorderLayout br=new BorderLayout();
      frm.setTitle("Dialog Demo");
      frm.setLayout(br);
      frm.setSize(200,150);
      frm.add(txa,br.CENTER);
      frm.add(btn,br.SOUTH);
      btn.addActionListener(frm);   // 設定frm為btn的事件傾聽者
      frm.setVisible(true);
   }
   // 當Open按鈕按下時，執行下列的程式碼
   public void actionPerformed(ActionEvent e)
   {
      fdlg.setVisible(true);        // 顯示檔案對話方塊
      // 以下的程式碼是按下檔案對話方塊內的「開啟」鈕之後，才會執行
      String fname=fdlg.getDirectory()+fdlg.getFile(); // 取得路徑與名稱
      try
      {
         FileInputStream fi=new FileInputStream(fname);
         byte ba[]=new byte[fi.available()];
         fi.read(ba);         // 讀入檔案內容到byte陣列裡
         txa.setText(new String(ba)); // 將byte陣列的內容寫到txa物件裡
         fi.close();
      }
      catch(IOException ioe){};
   }
}
