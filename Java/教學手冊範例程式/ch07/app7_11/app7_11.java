// app7_11, ノま计计ぃㄓ更method絛ㄒ
public class app7_11
{
   public static void main(String args[])
   {
      star();     // ㊣11~14︽┮﹚竡star() method
      star(7);       // ㊣16~21︽┮﹚竡star() method
      star('@',9);   // ㊣23~28︽┮﹚竡star() method
   }

   public static void star()     // ⊿Τま计star() method
   {
      star(5);       // ㊣16~21︽┮﹚竡star()肚俱计5
   }

   public static void star(int n)      // Τま计star() method
   {
      for(int i=0;i<n;i++)
         System.out.print("*");
      System.out.println();
   }

   public static void star(char ch, int n) // Τㄢま计star() method
   {
      for(int i=0;i<n;i++)
         System.out.print(ch);
      System.out.println();
   }
}