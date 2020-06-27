// app4_1, 設定運算子「=」
public class app4_1
{
   public static void main(String args[])
   {
      int age=21;    // 宣告整數變數age,並設值為21

      System.out.println("before compute, age="+age);  // 印出age的值
      age=age+1;     // 將age加1後再設定給age存放
      System.out.println("after compute, age="+age);  // 印出計算後age的值
   }
}

/* app4_1 OUTPUT------------
before compute, age=21
after compute, age=22
---------------------------*/