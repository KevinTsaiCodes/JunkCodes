// app19_8, 對話方塊的實例應用
import java.awt.*;
import java.awt.event.*;
public class app19_8 extends Frame implements ActionListener,ItemListener
{
   static app19_8 frm=new app19_8();
   static Dialog dlg=new Dialog(frm);                    // 建立Dialog物件dlg
   static Button Change_btn=new Button("Change Color");  // 按鈕
   static Button Ok_btn=new Button("Ok");
   static Button Cancel_btn=new Button("Cancel");
   static Checkbox ckb1=new Checkbox("White");           // 核取方塊
   static Checkbox ckb2=new Checkbox("Yellow");
   static Color clr;

   public static void main(String args[])
   {
      frm.setLayout(new FlowLayout(FlowLayout.CENTER,5,30));
      CheckboxGroup grp=new CheckboxGroup(); // 建立群組物件
      ckb1.setCheckboxGroup(grp);            // 將核取方塊加入群組
      ckb2.setCheckboxGroup(grp);
      frm.setSize(200,150);
      frm.setTitle("Frame");
      frm.setBackground(Color.pink);
      dlg.setTitle("Dialog"); // 設定對話方塊的標題
      dlg.setSize(150,120);   // 設定對話方塊的大小
      dlg.setLayout(null);
      ckb1.setBounds(20,50,60,20);  // 設定核取方塊的位置大小
      ckb2.setBounds(20,70,60,20);
      Ok_btn.setBounds(85,45,50,20);// 設定按鈕的位置大小
      Cancel_btn.setBounds(85,70,50,20);
      dlg.add(Ok_btn);        // 將按鈕加入對話方塊
      dlg.add(Cancel_btn);
      dlg.add(ckb1);          // 將核取方塊加入對話方塊
      dlg.add(ckb2);
      Change_btn.addActionListener(frm);  // 設定Change_btn的傾聽者
      Cancel_btn.addActionListener(frm);  // 設定Cancel_btn的傾聽者
      Ok_btn.addActionListener(frm);      // 設定Ok_btn的傾聽者
      ckb1.addItemListener(frm);          // 設定核取方塊ckb1的傾聽者
      ckb2.addItemListener(frm);          // 設定核取方塊ckb2的傾聽者
      frm.add(Change_btn);
      frm.setVisible(true);
   }

   public void actionPerformed(ActionEvent e)   // 按下對話方塊上的按鈕
   {
      Button btn=(Button) e.getSource();        // 取得被按下的按鈕
      if(btn==Change_btn)                       // 按下Change Color鈕
      {
         dlg.setLocation(120,80);               // 設定對話方塊的位置
         dlg.setVisible(true);                  // 顯示對話方塊
      }
      else if(btn==Ok_btn)                      // 如果是Ok鈕被按
      {
         frm.setBackground(clr);                // 設定背景顏色
         dlg.setVisible(false);                 // 隱藏對話方塊
      }
      else if(btn==Cancel_btn)                  // 如果是Cancel鈕被按
         dlg.setVisible(false);                 // 隱藏對話方塊
   }

   public void itemStateChanged(ItemEvent e)    // 選擇顏色
   {
      if(ckb1.getState())                       // ckb1被選取
         clr=Color.white;                       // 設成白色
      else if(ckb2.getState())                  // ckb2被選取
         clr=Color.yellow;                      // 設成黃色
   }
}
