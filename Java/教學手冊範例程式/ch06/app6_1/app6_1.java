// app6_1, 一維陣列
public class app6_1
{
   public static void main(String args[])
   {
      int i;
      int a[];       // 宣告整數陣列a
      a=new int[3];  // 配置可存放3個整數的記憶體空間供整數陣列a使用
      a[0]=9;     // 設定第一個元素的值為9
      a[1]=6;     // 設定第二個元素的值為6

      for(i=0; i<a.length; i++)               // 印出陣列的內容
         System.out.print("a["+i+"]="+a[i]+",\t");

      System.out.println("\nLength of array = "+a.length);   // 印出陣列長度
   }
}