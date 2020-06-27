// app12_10, StringBuffer類別使用的範例
public class app12_10
{
   public static void main(String args[])
   {
      StringBuffer str=new StringBuffer("Black & White");

      System.out.println(str);
      System.out.println("length="+str.length());
      System.out.println(str.replace(0,5,"cats"));
      System.out.println(str.replace(7,12,"dogs"));
      System.out.println(str.reverse());
      System.out.println(str);
   }
}

/* app12_10 OUTPUT-----
Black & White
length=13
cats & White
cats & dogs
sgod & stac
sgod & stac
---------------------*/