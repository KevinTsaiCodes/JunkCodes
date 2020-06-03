import java.io.IOException;
import java.io.PrintWriter;
import java.net.Socket;
import java.net.UnknownHostException;
import java.util.Scanner;

public class p02_client {
	public static void main(String[] args) throws UnknownHostException, IOException {
		Socket client = new Socket("127.0.0.1",8888);
		/* 這實體化是針對 client 端來做
			Socket 實體化名稱（客戶端） = new Socket("伺服器IPv4位址",通訊阜號);
		*/
		PrintWriter pw = new PrintWriter(client.getOutputStream());
		// getOutputStream 意思是說 將資料傳入緩衝區
		Scanner scn = new Scanner(System.in);
		System.out.println("Input a number: ");
		int n = scn.nextInt();
		pw.write(n); // 寫入緩衝區
		pw.flush(); // 清除緩衝區資料,類似 C 語言的 fflush(stdin);
	}
}
