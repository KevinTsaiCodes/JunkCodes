/*************************************************/
/*【程式名稱】: 5_fib_nr.c                       */
/*【程式功能】: 以非遞迴方式列印出費氏數列       */
/*       輸入 : 整數 n                           */
/*       輸出 : 列印出第 n 項費氏數              */
/*【資料結構】:                                  */
/*************************************************/

#include <stdio.h>

long fib_nr(long n);

/*-------------------------------------*/
/*   以非遞迴方式列印出第 n 項費氏數   */
/*-------------------------------------*/
long fib_nr(long n)
{
   long fn1, fn2, fn;
   int i;

   if(n == 0)
      return 0; 
   if(n == 1)
      return 1;
   if(n > 1){
      fn1 = 0;
      fn2 = 1;
      for(i = 2; i <= n; i++){
         fn = fn1 + fn2;
         fn1 = fn2;
         fn2 = fn;
      }
      return fn;
   }
   else 
      printf("\n錯誤! n 必須為大於 0 的整數!");
 }

void main(void)
{
   int i;

   clrscr( );
   printf("\n第 0 項至第 12 項費氏數為 : ");
   for(i=0; i <= 12; i++)
      printf("%ld  ",fib_nr(i));
}