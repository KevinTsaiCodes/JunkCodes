// app5_5, while 迴圈
public class app5_5
{
   public static void main(String args[])
   {
      int n=0,sum=0;
      while(sum<20)
      {
         System.out.println("n="+n+", sum="+sum);
         n++;       // 將n值加1
            sum+=n;    // 累加計算
      }
   }
}

/* app5_5 OUTPUT------
n=0, sum=0
n=1, sum=1
n=2, sum=3
n=3, sum=6
n=4, sum=10
n=5, sum=15
--------------------*/