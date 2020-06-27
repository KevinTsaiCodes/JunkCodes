// app16_2, 簡單的泛型範例
public class app16_2
{
   public static void main(String args[])
   {
      CMember<Integer> obj1=new CMember<Integer>();
      CMember<String>  obj2=new CMember<String>();
      obj1.setId(6);
      obj2.setId("Lily");
      obj1.show();
      obj2.show();
   }
}
class CMember<T>        // 定義泛型類別CMember，T為通用的型態
{
   private T id;        // 宣告 id的型態為T
   public void setId(T value)
   {
   	  id=value;       // 將id成員設為傳入的引數
   }
   public void show()
   {
   	  System.out.println("id="+id);
   }
}
