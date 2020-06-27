// app5_6, do while迴圈
import java.util.Scanner;
public class app5_6
{
   public static void main(String args[])
   {
      Scanner scn=new Scanner(System.in);
      int n,i=1,sum=0;

      do{
         System.out.print("請輸入累加的最大值: ");
         n=scn.nextInt();
      }while(n<1);      // 輸入n,n要大於等於1,否則會一直重複輸入

      do
         sum+=i++;      // 計算sum=sum+i,然後i值再加1
      while(i<=n);

      System.out.println("1+2+...+"+n+"="+sum);    // 印出結果
   }
}

/* app5_6 OUTPUT-------
請輸入累加的最大值: -8
請輸入累加的最大值: 10
1+2+...+10=55
---------------------*/