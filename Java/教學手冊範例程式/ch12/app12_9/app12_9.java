// app12_9, String類別使用的範例
public class app12_9
{
   public static void main(String args[])
   {
      String str="Easier said than done.";
      System.out.println("length="+str.length());
      System.out.println("charAt(8)="+str.charAt(8));
      System.out.println("sub string="+str.substring(7));
      System.out.println("start with \"th\"="+str.startsWith("th"));
      System.out.println("upper case="+str.toUpperCase());
   }
}

/* app12_9 OUTPUT-------------------
length=22
charAt(8)=a
sub string=said than done.
start with "th"=false
upper case=EASIER SAID THAN DONE.
----------------------------------*/