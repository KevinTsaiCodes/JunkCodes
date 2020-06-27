// app7_1, 簡單的範例
public class app7_1
{
  public static void main(String args[])
  {
     star();   // 呼叫star() method
     System.out.println("Knowledge is power");
     star();   // 呼叫star() method
  }

  public static void star()  // star() method
  {
     for(int i=0;i<20;i++)
        System.out.print("*");  // 印出20個星號
     System.out.print("\n");    // 換行
  }
}

/* app7_1 OUTPUT----
********************
Knowledge is power
********************
--------------------*/