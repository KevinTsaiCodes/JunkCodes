package FinalExam;
import java.io.IOException;
import java.io.PrintWriter;
import java.net.Socket;
import java.net.UnknownHostException;
import java.util.Scanner;

public class p04_client {
	
	public static void main(String[] args) throws UnknownHostException, IOException {
		Socket client = new Socket("localhost", 8888);
		PrintWriter pw = new PrintWriter(client.getOutputStream());
		Scanner scn = new Scanner(System.in);
		try {
			System.out.println("input a number:");
			int n = scn.nextInt();
			pw.write(n);
		}catch(Exception e) {
			System.err.println("Wrong inputx, input again...");
			main(args);
		}finally {
			System.out.println("End of Client Socket...");
			pw.close();
		}
	}
}
