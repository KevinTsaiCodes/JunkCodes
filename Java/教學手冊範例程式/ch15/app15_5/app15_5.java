// app15_5, 執行緒排程的設計(一)
class CTest extends Thread    // 從Thread類別延伸出子類別
{
   private String id;
   public CTest(String str)      // 建構元，設定成員id
   {
      id=str;
   }
   public void run()          // 改寫Thread類別裡的run() method
   {
      for(int i=0;i<4;i++)
      {
         try
         {
            sleep((int)(1000*Math.random()));
         }
         catch(InterruptedException e){}
         System.out.println(id+" is running..");
      }
   }
}

public class app15_5
{
   public static void main(String args[])
   {
      CTest dog=new CTest("doggy");
      CTest cat=new CTest("kitty");
      dog.start();            // 用dog物件來啟動執行緒
      cat.start();               // 用cat物件來啟動執行緒
      System.out.println("main() finished");
   }
}

/* app15_5 OUTPUT---------
main() finished
doggy is running..
kitty is running..
doggy is running..
doggy is running..
doggy is running..
kitty is running..
kitty is running..
kitty is running..
------------------------*/
