// app16_1, 簡單的範例
import java.util.*;
public class app16_1
{
   public static void main(String args[])
   {
      HashSet<String> hset=new HashSet<String>();

      hset.add("Monkey");     // 增加元素
      hset.add("Bunny");      // 增加元素
      hset.add("Monkey");     // 增加元素

      System.out.println("HashSet內容:"+hset); // 顯示集合物件的內容
   }
}


/* app16_1 OUTPUT------------
HashSet內容:[Bunny, Monkey]
---------------------------*/