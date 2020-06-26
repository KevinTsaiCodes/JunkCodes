package FinalExam;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.ServerSocket;
import java.net.Socket;

public class p04_server {
	
	public static void main(String[] args) throws IOException {
		try {
			ServerSocket server = new ServerSocket(8888);
			Socket client = server.accept();
			InputStreamReader isr = new InputStreamReader(client.getInputStream());
			BufferedReader br = new BufferedReader(isr);
			int n = br.read();
			System.out.println(n);
			br.close();
		}catch(Exception e) {
			System.err.println("Server do not connected!");
		}finally {
			System.out.println("End of Server Socket!");
		}
		
	}
}
