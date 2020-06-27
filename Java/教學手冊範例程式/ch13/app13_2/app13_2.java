// app13_2, 例外的處理
 public class app13_2
 {
   public static void main(String args[])
   {
      try         // 檢查這個程式區塊的程式碼
      {
         int arr[]=new int[5];
         arr[10]=7;
      }
      catch(ArrayIndexOutOfBoundsException e)
      {
         System.out.println("index out of bound!!");
      }
      finally        // 這個區塊的程式碼一定會執行
      {
         System.out.println("this line is always executed!!");
      }
      System.out.println("end of main() method!!");
   }
}

/* app13_2 OUTPUT----------------
index out of bound!!
this line is always executed!!
end of main() method!!
-------------------------------*/