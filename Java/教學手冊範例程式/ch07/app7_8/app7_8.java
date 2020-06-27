// app7_8,「傳參照」的範例
public class app7_8
{
   public static void main(String args[])
   {
      int A[]={1,2,3,4,5};

      square(A);                 // 呼叫 square()，並傳遞陣列A

      System.out.println("呼叫square() method之後...");

      for(int i=0;i<A.length;i++)     // 印出陣列的內容
         System.out.print(A[i]+" ");

      System.out.println();
   }

   public static void square(int arr[])
   {
      for(int i=0;i<arr.length;i++)
         arr[i]=arr[i]*arr[i];         // 將陣列的元素值平方
   }
}

/* app7_8 OUTPUT-----------
呼叫square() method之後...
1 4 9 16 25
-------------------------*/