// app15_6, 執行緒排程的設計(一)
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

public class app15_6
{
   public static void main(String args[])
   {
      CTest dog=new CTest("doggy");
      CTest cat=new CTest("kitty");

      dog.start();   // 啟動dog執行緒
      try
      {
         dog.join(); // 限制dog執行緒結束後才能往下執行
         cat.start();   // 啟動cat執行緒
         cat.join(); // 限制cat執行緒結束後才能往下執行
      }
      catch(InterruptedException e){}
      System.out.println("main() finished");
   }
}