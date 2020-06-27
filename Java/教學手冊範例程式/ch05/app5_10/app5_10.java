// app5_10, switch敘述
public class app5_10
{
   public static void main(String args[])
   {
      int a=50,b=20;
      char oper='*';

      switch (oper)
      {
         case '+':         // 印出a+b
            System.out.println(a+"+"+b+"="+(a+b));
            break;
         case '-':         // 印出a-b
            System.out.println(a+"-"+b+"="+(a-b));
            break;
         case '*':         // 印出a*b
            System.out.println(a+"*"+b+"="+(a*b));
            break;
         case '/':         // 印出a/b
            System.out.println(a+"/"+b+"="+((float)a/b));
            break;
         default:          // 印出字串
            System.out.println("Unknown expression!!");
     }
  }
}

/* app5_10 OUTPUT------------
50*20=1000
---------------------------*/