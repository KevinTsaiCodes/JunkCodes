// app16_4, 簡單的TreeSet範例
import java.util.*;
public class app16_4
{
   public static void main(String args[])
   {
      TreeSet<Integer> tset=new TreeSet<Integer>();

      for(int i=20;i>=2;i-=2)				  // 增加元素
         tset.add(i);

      System.out.println("元素個數="+tset.size());
      System.out.println("集合內容="+tset);     // 顯示集合物件的內容

      System.out.println("第一個元素="+tset.first());
      System.out.println("最後一個元素="+tset.last());
      System.out.println("介於6和14之間的集合="+tset.subSet(6,14));
      System.out.println("大於等於10的集合="+tset.tailSet(10));
      System.out.println("小於8的集合="+tset.headSet(8));
   }
}
