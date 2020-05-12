#include<stdio.h>
#include<stdlib.h>

struct _node {
	int data;
	struct _node* nextPtr;

};
typedef struct _node node;

node* CreateNode(int data)
{
	node* nextNode = (node*)malloc(sizeof(node));
	nextNode->data = data;
	nextNode->nextPtr = NULL;
	return nextNode;
}
void print(node* head)
{
	node* start;
	start = head;
	while (start != NULL) {
		printf("%d\n", start->data);
		start = start->nextPtr;
	}
}
void release(node *head)
{
	node* ptr;
	while (head != NULL)
	{
		ptr = head;
		head = head->nextPtr;
		free(ptr);
	}
}
int main()
{
	node* head;
	head = CreateNode(5);
	head->nextPtr = CreateNode(10);
	head->nextPtr->nextPtr = CreateNode(20);
	print(head);
	release(head);
	return EXIT_SUCCESS;
}
