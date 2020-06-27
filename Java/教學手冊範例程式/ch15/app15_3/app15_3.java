// app15_3, 實作Runnable介面來建立執行緒
class CTest implements Runnable     // 由CTest類別實作Runnable介面
{
   private String id;
   public CTest(String str)         // 建構元，設定成員id
   {
      id=str;
   }
   public void run()    // 詳細定義runnable() 介面裡的 run() method
   {
      for(int i=0;i<4;i++)
      {
         for(int j=0;j<100000000;j++); // 空迴圈，用來拖慢14行執行的速度
         System.out.println(id+" is running..");
      }
   }
}

public class app15_3
{
   public static void main(String args[])
   {
      CTest dog=new CTest("doggy");
      CTest cat=new CTest("kitty");
      Thread t1=new Thread(dog);    // 產生Thread類別的物件t1
      Thread t2=new Thread(cat);    // 產生Thread類別的物件t2
      t1.start();             // 用t1啟動執行緒
      t2.start();             // 用t2啟動執行緒
   }
}

/* app15_3 OUTPUT---------
kitty is running..
doggy is running..
kitty is running..
kitty is running..
doggy is running..
kitty is running..
doggy is running..
doggy is running..
-----------------------*/
