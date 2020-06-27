// app11_8, instanceof運算子的使用
class CCircle { }
class CCoin extends CCircle { }     // CCoin繼承CCircle類別
public class app11_8 extends CCoin     // app11_8繼承CCoin類別
{
   public static void main(String args[])
   {
      boolean status;
      CCoin coin=new CCoin();
      CCircle cir=new CCircle();
      app11_8 myobj=new app11_8();
      CCoin carr[]=new CCoin[5];

      // 判別cir是否為CCoin類別或其子類別物件
      status=(cir instanceof CCoin);
      System.out.println("cir instanceof CCoin? " + status);

      // 判別myobj是否為CCircle類別或其子類別物件
      status=(myobj instanceof CCircle);
      System.out.println("myobj instanceof CCircle? " + status);

      // 判別coin是否為app11_8類別或其子類別物件
      status=(coin instanceof app11_8);
      System.out.println("coin instanceof app11_8? "+ status);

      // 判別coin是否為CCircle類別或其子類別物件
      status=(coin instanceof CCircle);
      System.out.println("coin instanceof CCircle? " + status);

      // 判別coin是否為CCoin類別或其子類別物件
      status=(coin instanceof CCoin);
      System.out.println("coin instanceof CCoin? " + status);

      // 判別陣列是否為Object類別或其子類別物件
      status=(carr instanceof Object);
      System.out.println("carr instanceof Object? " + status);

      // 判別cir是否為String類別或其子類別物件
      // status=(cir instanceof String);
      // System.out.println("cir instanceof String? "+ status);
  }
}

/* app11_8 OUTPUT---------------
cir instanceof CCoin? false
myobj instanceof CCircle? true
coin instanceof app11_8? false
coin instanceof CCircle? true
coin instanceof CCoin? true
carr instanceof Object? true
-----------------------------*/
