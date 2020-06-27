// app7_2, 簡單的範例--method的引數與傳回值
public class app7_2
{
   public static void main(String args[])
  {
     int i;    // 宣告整數變數i，此變數的有效範圍僅止於main() method
     i=star(9);   // 傳入9給star()，並以i接收傳回的數值
     System.out.println(i + " stars printed");
  }

  public static int star(int n)  // star() method
  {
     int i;    // 宣告整數變數i，此變數的有效範圍僅止於star() method
     for(i=1;i<=2*n;i++)
        System.out.print("*");   // 印出2*n個星號
     System.out.print("\n");     // 換行
     return 2*n;              // 傳回整數2*n
  }
}

/* app7_2 OUTPUT------
******************
18 stars printed
--------------------*/