import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.Scanner;

public class p01_jdbc_test {
	static final String JDBC_DRIVER = "com.mysql.jdbc.Driver";
	static final String DATABASE_URL = "jdbc:mysql://140.131.152.25/oop1082a";
	public static void main(String[] args){
		String acc, pwd, name, sid, sql;
		Connection con = null;
		Statement stmt = null;
		try {
			Class.forName(JDBC_DRIVER);
			con = DriverManager.getConnection(DATABASE_URL,"oop1082a","oop1082a");
			stmt = con.createStatement();
			Scanner scn = new Scanner(System.in);
			acc = scn.next();
			pwd = scn.next();
			name = scn.next();
			sid = scn.next();
			sql = String.format("INSERT INTO userlogin" + "VALUES('%s','%s','%s','%s')", acc, pwd, name, sid);
			System.out.print(sql);
			stmt.executeUpdate(sql);
		}catch(Exception e) {
			e.printStackTrace();
		}
	}
}