// app7_4, method傳值的範例
public class app7_4
{
   public static void main(String args[])
  {
     int num=5;
     add10(num);        // 呼叫add10(),並傳遞num
     System.out.println("in main(), num = "+num);
  }

  public static void add10(int value)
  {
     value=value+10;    // 將value的值加10之後，設回給value
     System.out.println("in add10(), value = "+value);
  }
}

/* app7_4 OUTPUT--------
in add10(), value = 15
in main(), num = 5
----------------------*/