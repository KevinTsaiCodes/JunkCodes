#include<cstdio>
#include<cstdlib>

int main()
{
	int score;
	scanf_s("%d", &score);
	if (score > 60 && score < 101)
		score += 10;
	else if (score > 100 || score < 0) {
		printf_s("error\n");
		exit(0);
	}
	else
		score += 5;
	printf("%d\n", score);
	system("pause");
	return 0;
}
