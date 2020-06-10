import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;


public class p02_print_writer {
	public static void main(String[] args) throws FileNotFoundException {
		String path = "pokemon_data.csv";
		File file = new File(path);
		System.out.println(file.exists());
		PrintWriter pw = null;
		if(!file.exists()) { // 不管有無檔案都會幫你建立檔案
			pw = new PrintWriter(new OutputStreamWriter(new FileOutputStream(file, true)));
		/*  pw = new PrintWriter(new OutputStreamWriter(new FileOutputStream(file, true)));
			表示檔案不會因每次跑 JVM 而覆蓋原本檔案內容，預設為 false，表示會被覆蓋
		*/
		}
		pw = new PrintWriter(new OutputStreamWriter(new FileOutputStream(file, true)));
		pw.write("Today"+","+"Yesterday"+","+"Tomorrow"+"\n");
		pw.flush(); // 清除緩衝區資料，也就是 IO
		pw.close();
	}
}
