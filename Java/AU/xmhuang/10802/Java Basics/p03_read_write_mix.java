import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.Scanner;

public class p03_read_write_mix {
	public static void main(String[] args) throws IOException {
		String path = "pokemon_data.csv";
		File file = new File(path);
		System.out.println(file.exists());
		PrintWriter pw = null;
		if(!file.exists()) { // 不管有無檔案都會幫你建立檔案
			pw = new PrintWriter(new OutputStreamWriter(new FileOutputStream(file, true)));
		/*  pw = new PrintWriter(new OutputStreamWriter(new FileOutputStream(file, true)));
			表示檔案不會因每次跑 JVM 而覆蓋原本檔案內容，預設為 false，表示會被覆蓋
		*/
		}
		pw = new PrintWriter(new OutputStreamWriter(new FileOutputStream(file, true)));
		Scanner scn = new Scanner(System.in);
		System.out.println("1.Input\n2.Output\n3.Exit");
		int op = scn.nextInt();
		String content = "";
		while(op!=3) {
			if(op == 1) {
				content = scn.next();
				pw.write(content+","+"\n");
				pw.flush();
			}
			else if(op == 2) {
				BufferedReader br = new BufferedReader(new FileReader(path)); // 開檔
				String s; 
				s = br.readLine(); // 讀一行
				String[] text;
				while((s = br.readLine()) != null) {
					text = s.split(",");
					System.out.println(s);
				}
				text = null;
				br.close();
			}
			System.out.println("1.Input\n2.Output\n3.Exit");
			op = scn.nextInt();
		}
		pw.close();
	}
}
