#include <stdio.h>
#include <stdlib.h>
#define StackSize 5 /* 設定"堆疊"大小為 5 */
int stack[StackSize]; /* 製作 "堆疊" */
int top = -1; /* top 紀錄目前"堆疊"頂端索引值，以(-1)代表"堆疊"為空 */
void Push(int data); /* 將資料加入"堆疊"的函式 */
int Pop(void); /* 將資料從"堆疊"取出來的函式 */
void PrintStack(void); /* 印出"堆疊"裡面內容 */

int main()
{
	int op,data;
	while (1)
	{
		puts("Using POP and PUSH to understanding how \"STACK\" works\n");
		puts("1. push");
		puts("2. pop");
		puts("3. print all");
		puts("4. end");
		printf("> ");
		scanf_s("%d", &op);
		switch (op) {
			case 1:
				printf_s("input data\n");
				scanf_s("%d", &data);
				Push(data);
				break;
			case 2:
				printf_s("pop data %d from the stack\n", Pop());
				break;
			case 3:
				PrintStack();
				break;
			case 4:
				return EXIT_SUCCESS;
				break;
		}
		system("pause");
		system("cls");
	}
	
}
void Push(int data)
{
	if (top == StackSize-1)
		printf("Stack 已滿\n");
	else
		stack[++top] = data;
}
int Pop(void)
{
	if (top == -1)
		puts("Stack 是空的");
	else
		return stack[top--]; /* 意思等同於 return stack[top]; top=top-1; */
}
void PrintStack(void)
{
	if (top == -1)
		puts("Stack 是空的");
	else {
		int i;
		for (i = 0; i <= top; i++)
			printf("%d",stack[i]);
	}
	puts("");
}
