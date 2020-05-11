package Sever;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.Scanner;

public class p01_sever_test {
	public static void main(String[] args) throws IOException {
		ServerSocket ser_socket = new ServerSocket(8888);
		Socket cli_soc = ser_socket.accept();
		/*
			 Listens for a connection to be made to this socket and accept.
			 The method blocks until a connection is made. 
			 A new Socket s is created and, if there is a security manager,
			 the security manager's checkAccept method is called with s.getInetAddress().getHostAddress() 
			 and s.getPort()as its arguments to ensure the operation is allowed.This could result in
			 a SecurityException.
			 
			 accept method, accept the message from the client
		*/
		System.out.println("Receiving message from Client...");
		InputStreamReader isr = new InputStreamReader(cli_soc.getInputStream());
		BufferedReader br = new BufferedReader(isr);
		int n = br.read();
		System.out.println(n);
	}
}
/*
 * Remember the Server Socket and the Client Socket need same port
 * number, otherwise the socket cannot be connected!
 */
