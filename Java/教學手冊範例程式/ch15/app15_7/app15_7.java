// app15_7, 執行緒的優先順序
class CTest extends Thread       // 從Thread類別延伸出子類別
{
   private String id;
   public CTest(String str)      // 建構元，設定成員id
   {
      id=str;
   }
   public void run()             // 改寫Thread類別裡的run() method
   {
      for(int i=0;i<3;i++)
      {
         try
         {
            sleep(1000);
         }
         catch(InterruptedException e){}
         System.out.println(id+" is running..Priority="+this.getPriority());
      }
   }
}

public class app15_7
{
   public static void main(String args[])
   {
      CTest dog=new CTest("doggy");
      CTest cat=new CTest("kitty");
      CTest rabbit=new CTest("rabbit");
      CTest sheep=new CTest("sheep");
      CTest horse=new CTest("horse");

      cat.setPriority(Thread.MAX_PRIORITY);       // 設定執行序的優先權
      dog.setPriority(Thread.MIN_PRIORITY);
      rabbit.setPriority(7);
      horse.setPriority(3);

      dog.start();               // 啟動執行緒
      cat.start();
      rabbit.start();
      sheep.start();
      horse.start();
   }
}

/* app15_7 OUTPUT--------------
kitty is running..Priority=10
sheep is running..Priority=5
rabbit is running..Priority=7
horse is running..Priority=3
doggy is running..Priority=1
sheep is running..Priority=5
kitty is running..Priority=10
rabbit is running..Priority=7
horse is running..Priority=3
doggy is running..Priority=1
kitty is running..Priority=10
sheep is running..Priority=5
rabbit is running..Priority=7
horse is running..Priority=3
doggy is running..Priority=1
-----------------------------*/
