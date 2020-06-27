// app16_3, 簡單的HashSet範例
import java.util.*;
public class app16_3
{
   public static void main(String args[])
   {
      HashSet<String> hset=new HashSet<String>();

      String str1="Puppy";
      String str2="Kitty";
      System.out.println("Hash empty: "+hset.isEmpty());
      hset.add("Monkey");		// 增加元素
      hset.add("Bunny");      	// 增加元素
      hset.add(str1);     		// 增加元素
      hset.add(str2);         	// 增加元素

      System.out.println("Hash size="+hset.size()); // 顯示元素個數
      System.out.println("Hash empty: "+hset.isEmpty());
      System.out.println("HashSet內容:"+hset); // 顯示集合物件的內容

      hset.remove(str2);
      System.out.println("清除Kitty..., Hash size="+hset.size());

      System.out.println("Hash中是否有"+str2+"? "+hset.contains(str2));
      System.out.println("Hash中是否有fish? "+hset.contains("fish"));
      System.out.println("Hash中是否有Puppy? "+hset.contains("Puppy"));
      hset.remove("Bunny");
      System.out.println("清除Bunny..., Hash size="+hset.size());

      System.out.println("HashSet內容:"+hset);
      hset.clear();
      System.out.println("清除Hash中所有的物件...");
      System.out.println("Hash empty: "+hset.isEmpty());
   }
}
