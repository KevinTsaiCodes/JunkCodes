package homework;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.UnknownHostException;
import java.util.Random;
import java.util.Scanner;
public class p01_Guess extends p01_Question_Server{ // Client... 
	static final int SIZE = 4;
	public static void main(String[] args) throws UnknownHostException, IOException {
		System.out.println("Let the Game start!\nGenerating the Question...");
		Scanner scn = new Scanner(System.in);
		int guess_num;
		System.out.print("Your guess\n? ");
		guess_num = scn.nextInt();
		Socket CS = new Socket("127.0.0.1", 4999);
		PrintWriter pw = new PrintWriter(CS.getOutputStream());
		pw.println(guess_num);
		pw.flush();
		InputStreamReader in = new InputStreamReader(CS.getInputStream());
		BufferedReader bf = new BufferedReader(in);
		String result_from_server = bf.readLine();
		System.out.println(result_from_server);
		pw.close();
	}
}
