// app16_8, TreeMap範例
import java.util.*;
public class app16_8
{
   public static void main(String args[])
   {
      int k1=94001,k2=94003,key;
      TreeMap<Integer,String> tmap=new TreeMap<Integer,String>();

      tmap.put(94001,"Fiona");
      tmap.put(94003,"Ariel");
      tmap.put(94002,"Ryan");
      tmap.put(94004,"Jack");

      System.out.println("元素個數="+tmap.size());
      System.out.println("TreeMap的元素:"+tmap);
      key=tmap.firstKey();
      System.out.println("第0個元素= "+key+", "+tmap.get(key));
      key=tmap.lastKey();
      System.out.println("最後一個元素= "+key+", "+tmap.get(key));
      System.out.print("介於"+k1+"和"+k2+"之間的TreeMap=");
      System.out.println(tmap.subMap(k1,k2));
      System.out.print("大於等於"+k2+"的TreeMap=");
      System.out.println(tmap.tailMap(k2));
   }
}
