// app11_9, instanceof運算子的使用
public class app11_9
{
   public static void main(String args[])
   {
      int a[]=new int[5];

      if(a instanceof Object)          // 判別整數陣列是否繼承Object
         System.out.println("整數陣列a繼承Object");
      else
         System.out.println("整數陣列a沒有繼承Object");

      if(a instanceof Object[])          // 判別整數陣列是否繼承Object[]
         System.out.println("整數陣列a繼承Object[]");
      else
         System.out.println("整數陣列a沒有繼承Object[]");
  }
}

/* app11_9 OUTPUT編譯器訊息----------------------------------------------
error: inconvertible types
      if(a instanceof Object[])          // 判別整數陣列是否繼承Object[]
         ^
  required: Object[]
  found:    int[]
-----------------------------------------------------------------------*/
