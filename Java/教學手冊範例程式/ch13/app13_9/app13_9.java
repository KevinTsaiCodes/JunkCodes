// app13_9, 捕捉InputMismatchException例外
import java.util.*;
public class app13_9
{
   public static void main(String args[])
   {
      int num;
      Scanner scn=new Scanner(System.in);
      try
      {
         System.out.print("請輸入一個整數: ");
         num=scn.nextInt();
      }
      catch(Exception e)
      {
         System.out.println("拋出"+e+"例外");
      }
   }
}

/* app13_9 OUTPUT-------------------------
請輸入一個整數: k
拋出java.util.InputMismatchException例外
----------------------------------------*/