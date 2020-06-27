// app10_10, 設定終止繼承
class Caaa
{
   public final void show()   // 父類別的show()已被設為終止繼承
   {
      System.out.println("show() method in class Caaa called");
   }
}
class Cbbb extends Caaa
{
   public void show()         // 錯誤，改寫父類別的show() method
   {
      System.out.println("show() method in class Cbbb called");
   }
}
public class app10_10
{
   public static void main(String args[])
   {
      Cbbb b=new Cbbb();
      b.show();
   }
}

/* app10_10 OUTPUT編譯器訊息---------------------------------------
error: show() in Cbbb cannot override show() in Caaa
   public void show()         // 錯誤，改寫父類別的show() method
               ^
  overridden method is final
-----------------------------------------------------------------*/
