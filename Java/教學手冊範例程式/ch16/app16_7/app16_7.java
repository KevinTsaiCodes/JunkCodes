// app16_7, HashMap範例
import java.util.*;
public class app16_7
{
   public static void main(String args[])
   {
      HashMap<Integer,String> hmap=new HashMap<Integer,String>();

      hmap.put(94001,"Fiona");
      hmap.put(94003,"Ariel");
      hmap.put(94002,"Ryan");

      System.out.println("元素個數="+hmap.size());
      System.out.println("HashMap的元素:"+hmap);
      System.out.print("HashMap中是否有關鍵值94002? ");
      System.out.println(hmap.containsKey(94002));
      System.out.print("HashMap中是否有對應值Kevin? ");
      System.out.println(hmap.containsValue("Kevin"));
      hmap.remove(94001);
      System.out.print("清除關鍵值94001的資料..., ");
      System.out.println("元素個數="+hmap.size());
      System.out.println("HashMap的元素:"+hmap);
      System.out.println("關鍵值94003的對應值="+hmap.get(94003));
   }
}
