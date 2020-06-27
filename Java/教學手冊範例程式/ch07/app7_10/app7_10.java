// app7_10, 引數型態不同的函數多載
public class app7_10
{
   public static void main(String args[])
   {
      int a=5, b[]={1,2,3,4};
      show(a);                   // 將整數a傳遞到show()裡
      show(b);                   // 將整數陣列b傳遞到show()
   }

   public static void show(int i)         // 定義show(),可接收整數變數
   {
      System.out.println("value= "+i);
   }

   public static void show(int arr[])  // 定義show(),可接收整數陣列
   {
      System.out.print("array=");
      for(int i=0;i<arr.length;i++)
          System.out.print(" "+arr[i]);
      System.out.println();
   }
}

/* app7_10 OUTPUT---
value= 5
array= 1 2 3 4
------------------*/
