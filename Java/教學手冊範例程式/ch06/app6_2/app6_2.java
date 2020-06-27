// app6_2, 一維陣列的設值
public class app6_2
{
   public static void main(String args[])
   {
      int sum=0;
      int a[]={62,7,12,3,8,47};     // 宣告整數陣列a,並設定初值

      for(int i=0;i<a.length;i++)      // 計算陣列元素的和
         sum+=a[i];

      System.out.println("Average "+(float)sum/a.length);
   }
}

/* app6_2 OUTPUT----
Average 23.166666
------------------*/