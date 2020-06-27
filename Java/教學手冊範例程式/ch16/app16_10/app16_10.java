// app16_10, 以Iterator走訪TreeSet元素
import java.util.*;
public class app16_10
{
   public static void main(String args[])
   {
      TreeSet<String> tset=new TreeSet<String>();
      String str="";
      tset.add("Monkey");     			// 增加元素
      tset.add("Bunny");      			// 增加元素
      tset.add("Puppy");				// 增加元素
      tset.add("Kitty");    			// 增加元素

      Iterator<String> itr=tset.iterator(); // 取得Iterator物件
      System.out.print("TreeSet內容:");
      while(itr.hasNext())				// 走訪元素
      {
         str=itr.next();
         System.out.print(str+" ");		// 印出元素內容
      }

      System.out.println("\n刪除最後讀取的元素"+str+"...");
      itr.remove();					// 刪除最後讀取的元素
      System.out.println("TreeSet內容:"+tset);
   }
}
