import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.ServerSocket;
import java.net.Socket;

public class p02_server {
	public static void main(String[] args) throws IOException {
		ServerSocket ser_socket = new ServerSocket(8888);
		/* 這實體化是針對 server 端來做
			ServerSocket 實體化名稱（伺服器端） = new ServerSocket(通訊阜號);
			Client 與 Server 阜號 要一致才能連線喔
		 */
		Socket cli_soc = ser_socket.accept();
		// 客戶端必須伺服器端同意(accept)
		System.out.println("Receiving message from Client...");
		InputStreamReader isr = new InputStreamReader(cli_soc.getInputStream());
		// InputStreamReader： 讀取串流資料; getInputputStream 意思是說 將資料從緩衝區讀出來
		BufferedReader br = new BufferedReader(isr);
		int n = br.read();
		System.out.println(++n);
	}
}
