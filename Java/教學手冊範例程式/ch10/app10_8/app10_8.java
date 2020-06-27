// app10_8, 透過super關鍵字來存取父類別的變數
class Caaa
{
   protected int num;            // 父類別的資料成員num

   public void show()
   {
      System.out.println("Caaa_num="+num);
   }
}
class Cbbb extends Caaa
{
   int num=10;                      // 子類別的資料成員num

   public void show()
   {
      super.num=20;                 // 設定父類別的資料成員num為20
      System.out.println("Cbbb_num="+num);
      super.show();              // 呼叫父類別的show() method
   }
}

public class app10_8
{
   public static void main(String args[])
   {
      Cbbb b=new Cbbb();
      b.show();
   }
}

/* app10_8 OUTPUT---
Cbbb_num=10
Caaa_num=20
------------------*/