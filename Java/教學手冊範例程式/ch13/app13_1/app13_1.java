// app13_1, 索引值超出範圍
public class app13_1
{
   public static void main(String args[])
   {
      int arr[]=new int[5];            // 容許5個元素
      arr[10]=7;                       // 索引值超出容許範圍
      System.out.println("end of main() method !!");
   }
}