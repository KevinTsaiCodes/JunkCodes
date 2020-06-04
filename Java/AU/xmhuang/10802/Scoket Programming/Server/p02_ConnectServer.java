package Practice;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.ServerSocket;
import java.net.Socket;

public class p02_ConnectServer {
	public static void main(String[] args) throws IOException {
		int port = 1234;
		ServerSocket server = new ServerSocket(port);
		Socket client = server.accept();
		InputStreamReader isr = new InputStreamReader(client.getInputStream());
		BufferedReader br = new BufferedReader(isr);
		int question = br.read();
		int guess = br.read();
		if(question < guess)
			System.out.println("Too Large!");
		else if(question > guess)
			System.out.println("Too Small!");
		else
			System.out.println("Correct!");
		br.close();
		System.out.println("Answer: " +question);
	}
}
