package Practice;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

public class p01_buffered_reader {
	public static void main(String[] args) throws IOException {
		String path = "pokemon_data.csv";
		BufferedReader br = new BufferedReader(new FileReader(path)); // 開檔
		String s; 
		s = br.readLine(); // 讀一行
		while((s = br.readLine()) != null) {
			System.out.println(s);
		}
		br.close(); // 關檔
	}
}
