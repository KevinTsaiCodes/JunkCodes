import java.util.Scanner;

public class p04_my_func {
	public static float my_func(float x, int n) {
		int sum = 0;
		for(int i=1;i<=n;i++) {
			sum += Math.pow(x, i);
		}
		return sum;
	}
	public static void main(String[] args) {
		Scanner scn = new Scanner(System.in);
		float x = scn.nextFloat();
		int n = scn.nextInt();
		System.out.println(my_func(x,n));
	}
}
