// App21_9, 在applet播放音樂檔
import java.awt.*;
import java.awt.event.*;
import java.applet.Applet;
import java.applet.AudioClip;     //  載入AudioClip類別

public class App21_9 extends Applet implements ItemListener
{
   AudioClip midi[]=new AudioClip[3]; // 宣告AudioClip介面型態的陣列
   AudioClip current;       // 宣告AudioClip介面型態的變數current

   Choice chc=new Choice();           // 建立Choice元件

   public void init()
   {
      midi[0]=getAudioClip(getCodeBase(),"小叮噹.midi");
      midi[1]=getAudioClip(getCodeBase(),"科學小飛俠.midi");
      midi[2]=getAudioClip(getCodeBase(),"無敵鐵金剛.midi");
      chc.add("小叮噹");
      chc.add("科學小飛俠");
      chc.add("無敵鐵金剛");
      add(chc);
      chc.addItemListener(this);    // 把applet當成chc的傾聽者
      current=midi[0];           // 設定目前播放的歌曲為midi[0]
      current.play();            // 播放歌曲
   }

   public void itemStateChanged(ItemEvent e)
   {
      current.stop();                  // 停止播放歌曲
      int index=chc.getSelectedIndex();      // 取得被選取的索引值
      current=midi[index];       // 設定播放的歌曲為midi[index]
      current.play();         // 播放歌曲
   }
}
