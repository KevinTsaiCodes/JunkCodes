// app15_8, 沒有同步處理的執行緒
class CBank
{
   private static int sum=0;
   public static void add(int n)
   {
      int tmp=sum;
      tmp=tmp+n;            // 累加匯款總額
      try
      {
         Thread.sleep((int)(1000*Math.random()));   // 小睡0~1秒鐘
      }
      catch(InterruptedException e){}
      sum=tmp;
      System.out.println("sum= "+sum);
  }
}
class CCustomer extends Thread // CCustomer類別，繼承自Thread類別
{
   public void run()        // run() method
   {
      for(int i=1;i<=3;i++)
         CBank.add(100);       // 將100元分三次匯入
   }
}
public class app15_8
{
   public static void main(String args[])
   {
      CCustomer c1=new CCustomer();
      CCustomer c2=new CCustomer();
      c1.start();
      c2.start();
   }
}

/* app15_8 OUTPUT----(沒有加synchronized的執行結果)
sum= 100
sum= 100
sum= 200
sum= 300
sum= 200
sum= 300
--------------------*/
