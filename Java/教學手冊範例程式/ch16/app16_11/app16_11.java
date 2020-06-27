// app16_11, 以ListIterator走訪LinkedList元素
import java.util.*;
public class app16_11
{
   public static void main(String args[])
   {
      LinkedList<Integer> llist=new LinkedList<Integer>();
      llist.add(5); 	//加入元素5
      llist.add(7); 	//加入元素7
      llist.add(10); 	//加入元素10
      llist.add(14); 	//加入元素13
      ListIterator<Integer> litr1=llist.listIterator();

      System.out.print("正向列出LinkedList內容:");
      while(litr1.hasNext())    				// 正向走訪元素
         System.out.print(litr1.next()+" ");    	// 印出元素內容
      System.out.println();

      ListIterator<Integer> litr2=llist.listIterator(llist.size());
      System.out.print("反向列出LinkedList內容:");
      while(litr2.hasPrevious())				// 反向走訪元素
         System.out.print(litr2.previous()+" ");	// 印出元素內容
      System.out.println();
   }
}
