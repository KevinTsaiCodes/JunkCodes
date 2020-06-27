// app15_2, 啟動執行緒的範例
class CTest extends Thread    // 從Thread類別延伸出子類別CTest
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
         for(int j=0;j<100000000;j++); // 空迴圈，用來拖慢14行執行的速度
         System.out.println(id+" is running..");
      }
   }
}

public class app15_2
{
   public static void main(String args[])
   {
      CTest dog=new CTest("doggy");
      CTest cat=new CTest("kitty");
      dog.start();            // 注意是呼叫start(),而不是run()
      cat.start();            // 注意是呼叫start(),而不是run()
   }
}

/* app15_2 OUTPUT-------
kitty is running..
doggy is running..
kitty is running..
kitty is running..
doggy is running..
kitty is running..
doggy is running..
doggy is running..
----------------------*/