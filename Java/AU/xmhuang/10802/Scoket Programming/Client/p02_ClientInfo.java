package Practice;

import java.io.IOException;
import java.io.PrintWriter;
import java.net.Socket;
import java.net.UnknownHostException;
import java.util.Random;
import java.util.Scanner;

public class p02_ClientInfo {
	public static void main(String[] args) throws UnknownHostException, IOException {
		String host = "localhost";
		int port = 1234;
		Socket client = new Socket(host,port);
		PrintWriter pw = new PrintWriter(client.getOutputStream());
		int question, guess;
		Random ran = new Random();
		question = ran.nextInt(100)+1;
		System.out.println("Answer: " +question);
		System.out.print("²q¼Æ¦r(½d³ò1~100)\n? ");
		Scanner scn = new Scanner(System.in);
		guess = scn.nextInt();
		pw.write(question);
		pw.write(guess);
		pw.flush();
		pw.close();
	}
}
