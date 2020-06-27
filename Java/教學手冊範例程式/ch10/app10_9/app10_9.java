// app10_9, 用this來呼叫實例變數
class Caaa
{
   public int num=10;    // num是實例變數

   public void show()
   {
      int num=5;         // num是區域變數，其有效範圍僅限於在show()內
      System.out.println("this.num="+this.num); // 印出實例變數
      System.out.println("num="+num);           // 印出區域變數
   }
}
public class app10_9
{
   public static void main(String args[])
   {
      Caaa a=new Caaa();
      a.show();
   }
}

/* app10_9 OUTPUT---
this.num=10
num=5
------------------*/

