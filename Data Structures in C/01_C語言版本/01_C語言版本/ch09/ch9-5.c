#define MAX 10
int A[MAX]={0};
int main(void) { 
    int  A[MAX] = {0,1,2,3,4,5,6,7,8,9}; 
    int i, find; 
   printf("===============程式描述================\n");
   printf("= 程式名稱：ch9-5.c                   =\n");
   printf("= 程式目的：內插搜尋法                =\n"); 
   printf("=======================================\n"); 
   printf("===========輸入(預先排序)==============\n");
   for(i = 0; i < MAX; i++) 
        printf("%d ", A[i]); 
    printf("\n請輸入數字(0~9)："); 
    scanf("%d", &find); 
   printf("================輸出===================\n");     
    if((i = intsrch(A, find)) >= 0) 
        printf("該數字是排在第 %d個順位 ", i+1); 
    else 
        printf("\n找不到!"); 
    
    printf("\n"); 
    system("PAUSE");
    return 0; 
} 
//內插搜尋法
int intsrch(int A[], int find) { 
    int low, mid, high; 

    low = 0; 
    high = MAX - 1; 

    while(low <= high) { 
        mid = low+ (find-A[low])/(A[high]-A[low])*(high-low); 
        if(mid < low || mid > high) 
            return -1; 
        if(find < A[mid]) 
            high = mid - 1; 
        else if(find > A[mid]) 
            low = mid + 1; 
        else 
            return mid; 
     } 
     return -1;
} 
