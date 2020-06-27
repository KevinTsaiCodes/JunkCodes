// app6_3, 比較陣列元素值的大小
public class app6_3
{
   public static void main(String args[])
   {
      int i,min,max;
      int a[]={74,48,30,17,62};  // 宣告整數陣列a,並設定初值

      min=max=a[0];
      System.out.print("Elements in array a are ");
      for(i=0;i<a.length;i++)
      {
         System.out.print(a[i]+"  ");
         if(a[i]>max)            // find Maximum
            max=a[i];
         if(a[i]<min)            // find Minimum
            min=a[i];
      }
      System.out.println("\nMaximum is "+max);  // 印出最大值
      System.out.println("Minimum is "+min);    // 印出最小值
   }
}

/* app6_3 OUTPUT----------------------------
Elements in array a are 74  48  30  17  62
Maximum is 74
Minimum is 17
------------------------------------------*/
