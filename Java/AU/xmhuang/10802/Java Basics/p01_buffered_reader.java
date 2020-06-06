import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

public class p01_buffered_reader {
	public static void main(String[] args) throws IOException {
		String path = "pokemon_data.csv";
		File file = new File(path);
		System.out.println("檔案是否存在: " +file.exists()); //驗證檔案用 
		BufferedReader br = new BufferedReader(new FileReader(path)); // 開檔
		String s; 
		s = br.readLine(); // 讀一行
		while((s = br.readLine()) != null) {
			System.out.println(s);
		}
		br.close(); // 關檔
	}
}
