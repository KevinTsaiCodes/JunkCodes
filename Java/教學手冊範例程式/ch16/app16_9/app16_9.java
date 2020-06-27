// app16_9, 以for-each loop走訪TreeSet元素
import java.util.*;
public class app16_9
{
   public static void main(String args[])
   {
      TreeSet<String> tset=new TreeSet<String>();
      tset.add("Monkey");           // 增加元素
      tset.add("Bunny");
      tset.add("Puppy");
      tset.add("Kitty");
      System.out.print("TreeSet內容:");

      for(String i:tset)            // 走訪
         System.out.print(i+" ");
   }
}
