// app7_3, 計算長方形對角線的長度
public class app7_3
{
   public static void main(String args[])
  {
     double num;
     num=show_length(8,4);  // 傳入8與4兩個引數到show_length()裡
     System.out.println("length = "+num);
  }

  public static double show_length(int m, int n)
  {
     return Math.sqrt(m*m+n*n);  // 傳回對角線長度
  }
}

/* app7_3 OUTPUT-----------
length = 8.94427190999916
-------------------------*/