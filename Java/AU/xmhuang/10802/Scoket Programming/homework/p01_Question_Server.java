package homework;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.UnknownHostException;
import java.util.Random;
public class p01_Question_Server {// Server...
	static final int SIZE = 4;
	public static void main(String[] args) throws UnknownHostException, IOException  {
		int[] question = new int[SIZE];	
		Random rand = new Random();
		for(int i=SIZE-1;i>=0;i--)
			question[i] = rand.nextInt(10);
		ServerSocket SS = new ServerSocket(4999);
		Integer a = new Integer(0), b = new Integer(0);
		Socket CS = SS.accept();
		System.out.println("Client connected!");
		InputStreamReader in = new InputStreamReader(CS.getInputStream());
		BufferedReader bf = new BufferedReader(in);
		int guess = Integer.parseInt(bf.readLine());
		int[] guess_num = new int[4];
		for(int i=0;i<SIZE;i++) {
			guess_num[i] = guess%10;
			guess/=10;
		}
		PrintWriter pw = new PrintWriter(CS.getOutputStream());
		for(int i=0;i<SIZE;i++) {
			if(question[i] == guess_num[i])
				a++;
			else {
				for(int j=0;j<SIZE;j++) {
					if(question[i] == guess_num[j])
						b++;
				}
			}
		}
		String result = a+"A"+b+"B";
		pw.println(result);
		pw.flush();
		pw.close();
	}
}