import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

public class p03_jdbc_test {
	static final String JDBC_DRIVER = "com.mysql.jdbc.Driver";
	static final String DATABASE_URL = "jdbc:mysql://localhost/art";
	public static void main(String[] args) {
		Connection con = null;
		Statement stmt = null;
		try {
			Class.forName(JDBC_DRIVER);
			con = DriverManager.getConnection(DATABASE_URL, "root", "");
			stmt = con.createStatement();
			ResultSet rs = stmt.executeQuery("SELECT * FROM artist");
			while (rs.next()) {
				System.out.println(rs.getString("ID")+ "  " + rs.getString("ARTIST_NAME"));
			}
		}catch(Exception e) {
			e.printStackTrace();
		}
	}
}
