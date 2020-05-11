package Client;
import java.io.IOException;
import java.io.PrintWriter;
import java.net.Socket;
import java.net.UnknownHostException;
import java.util.Scanner;


public class p01_client_test {
	public static void main(String[] args) throws UnknownHostException, IOException {
		/* 
		 * prints the stream to server, which get from the client.
		 * ※ prints it, so use (socket_name).getOutputStream
		*/
		Socket cli_soc = new Socket("127.0.0.1", 8888);
		PrintWriter pw = new PrintWriter(cli_soc.getOutputStream());
		Scanner scn = new Scanner(System.in);
		int n = scn.nextInt();
		System.out.println("Input a number: ");
		pw.write(n);
		pw.flush();
		pw.close();
		pw = null;
		cli_soc = null;
		scn = null;
	}
}

// Learning Video: https://youtu.be/-xKgxqG411c
