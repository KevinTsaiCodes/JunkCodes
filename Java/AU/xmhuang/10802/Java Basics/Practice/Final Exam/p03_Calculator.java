import java.util.Scanner;

public class p03_Calculator {
	public static void main(String[] args) {
		double a, b;
		System.out.println("請選擇運算子: + - * /");
		System.out.print("請選擇: ");
		Scanner scn = new Scanner(System.in);
		String operator = scn.next();
		if(operator.compareTo("+") == 0) {
			a = scn.nextDouble();
			b = scn.nextDouble();
			System.out.println(a+b);
		}
		else if(operator.compareTo("-") == 0) {
			a = scn.nextDouble();
			b = scn.nextDouble();
			System.out.println(a-b);
		}
		else if(operator.compareTo("*") == 0) {
			a = scn.nextDouble();
			b = scn.nextDouble();
			System.out.println(a*b);
		}
		else if(operator.compareTo("/") == 0) {
			a = scn.nextDouble();
			b = scn.nextDouble();
			double c = a/b;
			System.out.println(Math.round(c*10)/10);
		}
	}
}
