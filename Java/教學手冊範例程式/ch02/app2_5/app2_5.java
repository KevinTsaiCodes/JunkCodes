// app2_5,語意錯誤的程式
public class app2_5   // 定義public類別app2_5
{
   public static void main(String args[])
   {
      int num1=2;    // 宣告整數變數 num1，並設值為2
      int num2=3;    // 宣告整數變數 num2，並設值為3

      System.out.println("I have "+num1+" dogs");
      System.out.println("You have "+num2+" dogs");
      System.out.println("We have "+(num1-num2)+" dogs");
   }
}

/* app2_5 OUTPUT-----
I have 2 dogs
You have 3 dogs
We have -1 dogs
-------------------*/
