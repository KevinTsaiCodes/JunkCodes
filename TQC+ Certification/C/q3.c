#include<cstdio>
#include<cstdlib>
int compute(int score) 
{
	if (score > 60 && score < 101)
		score += 10;
	else if (score > 100 || score < 0)
		return -1;
	else
		score += 5;
	return score;
}
int main()
{
	int score;
	scanf_s("%d", &score);
	printf("%d\n", compute(score));
	system("pause");
	return 0;
}
