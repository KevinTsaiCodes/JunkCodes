import java.io.*;
import java.util.Scanner;
import java.lang.Math;

class Student {
	public String name;
	public String id;
	public short chinese;
	public short english;
	public short math;
	public Student() {
		this.name = "N/A";
		this.id = "N/A";
		this.chinese = 0;
		this.english = 0;
		this.math = 0;
	}
}

public class P03 {
	public static void main(String[] args)  throws IOException{
		Scanner scn = new Scanner(System.in);
		Student st = new Student(); // new class
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		File f1 = new File("score.csv"); // Open file
		PrintWriter pw = null;
		// Input the words
		if(!f1.exists()) {// If file not exists
			pw = new PrintWriter(new OutputStreamWriter(new FileOutputStream(f1))); // Create File
		}
		else {
			pw = new PrintWriter(new OutputStreamWriter(new FileOutputStream(f1))); // Create File
		}
		int op = 0, count = 0;
		float ch_avg=0f, eng_avg=0f, math_avg=0f;
		while(op!=4) {
			System.out.println("1. 輸入資料\n2. 讀入物件資料\n3. 印全班資料\n4. Exit");
			System.out.print("? ");
			op = scn.nextInt(); // Input Option
			if(op == 1) {
				st.name = br.readLine();
				st.id = br.readLine();
				st.chinese = scn.nextShort();
				st.english = scn.nextShort();
				st.math = scn.nextShort();
				pw.println(st.name + "," + st.id + "," + st.chinese+ "," + st.english + "," +st.math + ","
				+ (st.chinese+st.english+st.math) + "," + Math.round(((st.chinese+st.english+st.math)/3)*10.0)/10.0);
				++count;
				ch_avg+=(float)st.chinese;
				eng_avg+=(float)st.english;
				math_avg+=(float)st.math;
				pw.flush();
				
			}else if(op == 2) {
				Show(ch_avg,eng_avg,math_avg, count, f1);
			}else if(op == 3) {
				Show(ch_avg,eng_avg,math_avg, count, f1);
			}else if(op == 4) {
				System.out.print("Exit");
				break;
			}
			else
				System.err.println("Invalid Input! Input Again!");
			System.out.println();
		}
		
		pw.flush();
		pw.close();
	}
	public static void Show(float ch_avg, float eng_avg, float math_avg, int count, File f1) throws IOException {
		
		BufferedReader bfr  = new BufferedReader(new FileReader(f1));
		String data = ""; // Read String From File
		System.out.println("學生姓名\t學生學號\t國文成績\t英文成績\t數學成績\t總分\t平均");
		while((data = bfr.readLine()) != null) {// when is EOF end while loop
			String[] text = data.split(","); // string.split("要以啥作分割的符號");
			for(String space : text)
				System.out.print(space + "\t");
			System.out.println();
		}
		System.out.println("Average:\t" + Math.round(((ch_avg)/count)*10.0)/10.0+"\t" + Math.round(((eng_avg)/count)*10.0)/10.0+"\t" + Math.round(((math_avg)/count)*10.0)/10.0);
	}
}
