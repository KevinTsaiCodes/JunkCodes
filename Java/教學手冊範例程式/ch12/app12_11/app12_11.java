// app12_11, Integer class method的應用
public class app12_11
{
   public static void main(String args[])
   {
      String str;
      int inum;

      inum=Integer.parseInt("654")+3;     // 將字串轉成整數後，再加3
      System.out.println(inum);
      str=Integer.toString(inum)+"3";     // 將 "3" 附加在字串後面
      System.out.println(str);
   }
}

/* app12_11 OUTPUT---
657
6573
-------------------*/