// app5_9, continue的使用
public class app5_9
{
   public static void main(String args[])
   {
      int i;

      for (i=1;i<=10;i++)
      {
         if(i%3==0)              // 判斷i%3是否為0
            continue;
         System.out.println("i="+i);      // 印出i的值
      }
      System.out.println("when loop interruped,i="+i);
   }
}

/* app5_9 OUTPUT------------
i=1
i=2
i=4
i=5
i=7
i=8
i=10
when loop interruped,i=11
--------------------------*/