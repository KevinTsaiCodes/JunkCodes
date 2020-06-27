// app4_5, 邏輯運算子
public class app4_5
{
   public static void main(String args[])
   {
      int a=53;

      if ((a<0) || (a>100))
         System.out.println("Input error!!");    // 成績輸入錯誤
      if ((a<60) && (a>49))
         System.out.println("Make up exam!!");   // 需要補考
   }
}

/* app4_5 OUTPUT-----
Make up exam!!
-------------------*/