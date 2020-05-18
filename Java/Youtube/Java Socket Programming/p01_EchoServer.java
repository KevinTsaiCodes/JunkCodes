package Practice;

import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;

public class p01_EchoServer {
	
	public static void main(String[] args) {
		System.out.println("Waiting for Client...");
		try {
			ServerSocket SS = new ServerSocket(4999); // ServerSocket new_name = new ServerSocket(port_num);
			Socket socket = SS.accept(); /* Listens for a connection to be made to this socket and acceptsit.
											The method blocks until a connection is made. */
			System.out.println("Connection Established!");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
