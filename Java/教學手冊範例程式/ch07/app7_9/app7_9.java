// app7_9, 簡單的遞迴method
public class app7_9
{
   public static void main(String args[])
  {
     System.out.println("1*2*...*4="+fac(4));
  }
  public static int fac(int n)   // fac() method
  {
     if(n==0)                    // 設定終止條件
        return 1;
     else
        return n*fac(n-1);       // 遞迴計算
  }
}

/* app7_9 OUTPUT----
1*2*...*4=24
------------------*/