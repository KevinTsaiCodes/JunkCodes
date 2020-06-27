// app15_1, 單一執行緒的範例
class CTest
{
   private String id;
   public CTest(String str)            // 建構元，設定資料成員id
   {
      id=str;
   }
   public void run()                // run() method
   {
      for(int i=0;i<4;i++)
      {
         for(int j=0;j<100000000;j++);    // 空迴圈，用來拖慢14行執行的速度
         System.out.println(id+" is running..");
      }
   }
}

public class app15_1
{
   public static void main(String args[])
   {
      CTest dog=new CTest("doggy");
      CTest cat=new CTest("kitty");
      dog.run();
      cat.run();
   }
}