main( )
{   
    int i,j,count=0;
    int n=10;
    for(i=1;i<=n;i++)
      for(j=1;j<=n;j++)
        count=count+1;
    printf("count=%d\n",count);
    printf("i=%d\n",i);
    printf("j=%d\n",j);
    system("PAUSE");	
    return (0);
}
