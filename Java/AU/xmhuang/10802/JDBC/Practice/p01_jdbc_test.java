import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.Scanner;

public class p01_jdbc_test {
	static final String JDBC_DRIVER = "com.mysql.jdbc.Driver";
	static final String DATABASE_URL = "jdbc:mysql://localhost/artist";
	public static void main(String[] args) {
		String ID, Name, SQL;
		Connection con = null;
		Statement stmt = null;
		try {
			Class.forName(JDBC_DRIVER);
			con = DriverManager.getConnection(DATABASE_URL,"root","");
			stmt = con.createStatement();
			Scanner scn = new Scanner(System.in);
			ID = scn.next();
			Name = scn.next();
			SQL = String.format("INSERT INTO gallery" + " VALUES('%s','%s')", ID, Name);
			System.out.print(SQL);
			stmt.executeUpdate(SQL);
		}catch(Exception e) {
			e.printStackTrace();
		}
	}
}
