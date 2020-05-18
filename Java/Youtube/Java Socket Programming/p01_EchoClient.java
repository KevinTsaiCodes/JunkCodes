package Practice;

import java.io.IOException;
import java.net.Socket;

public class p01_EchoClient {
	
	public static void main(String[] args) {
		try {
			Socket socket = new Socket("localhost", 4999);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
