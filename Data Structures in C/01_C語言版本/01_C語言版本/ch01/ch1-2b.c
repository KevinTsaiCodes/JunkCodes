#define NUM 5
main( )
{   
    printf("===============程式描述=======================\n");
    printf("= 程式名稱：ch1-2b.c                         =\n");
    printf("= 程式目的：使用資料結構與演算法             =\n"); 
    printf("==============================================\n"); 
    float Average=0;
    int sum=0,i,Score[NUM]={60,70,80,85,90};  //宣告「陣列」資料結構 
    for (i = 0; i<NUM;i++ )                   //使用FOR迴圈的演算法 
       sum+=Score[i];
    Average=(float)(sum/NUM);
    printf("平均成績：%f\n",Average);
    system("PAUSE");	
    return 0;
}
