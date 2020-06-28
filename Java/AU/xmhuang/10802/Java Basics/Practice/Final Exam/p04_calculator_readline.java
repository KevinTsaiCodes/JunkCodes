package FinalExam;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class p04_calculator_readline {
	public static void main(String[] args) throws IOException {
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		System.out.println("Input Formula(Ex. 1+2=):");
		String formula = br.readLine(); // 因為 Scanner 遇到空白字元就會斷掉，所以不用Scanner
		if(formula.contains("+")) {
			String[] content = formula.split("\\+"); // 特殊符號得用 "\\"
			if (content.length > 1) {
				int num1 = Integer.parseInt(content[0].trim());
				int num2 = Integer.parseInt(content[1].substring(0,content[1].length()-1).trim());
				System.out.println(num1 + num2);
			}
		}
		else if(formula.contains("-")) {
			String[] content = formula.split("\\-"); // 特殊符號得用 "\\"
			if (content.length > 1) {
				int num1 = Integer.parseInt(content[0].trim());
				int num2 = Integer.parseInt(content[1].substring(0,content[1].length()-1).trim());
				System.out.println(num1 - num2);
			}
		}
		else if(formula.contains("*")) {
			String[] content = formula.split("\\*"); // 特殊符號得用 "\\"
			if (content.length > 1) {
				int num1 = Integer.parseInt(content[0].trim());
				int num2 = Integer.parseInt(content[1].substring(0,content[1].length()-1).trim());
				System.out.println(num1 * num2);
			}
		}
		else if(formula.contains("/")) {
			String[] content = formula.split("\\/"); // 特殊符號得用 "\\"
			if (content.length > 1) {
				int num1 = Integer.parseInt(content[0].trim());
				int num2 = Integer.parseInt(content[1].substring(0,content[1].length()-1).trim());
				System.out.println(num1 / num2);
			}
		}
	}
}
