import java.sql.*;
import java.util.Scanner;


public class p04_jdbc_test {
	static final String JDBC_DRIVER = "com.mysql.jdbc.Driver";
	static final String DATABASE_URL = "jdbc:mysql://localhost/livecoding";
	static String ID, Name, sql; // SQL is for SQL statement
	static int Books;
	
	public static void main(String[] args) {
		Connection con = null;
		Statement stmt = null;
		try {
			while(true) {
				Class.forName(JDBC_DRIVER);
				con = DriverManager.getConnection(DATABASE_URL, "root", "");
				stmt = con.createStatement();
				Scanner scn = new Scanner(System.in); // Input ID, Name, Books
				System.out.println("Input Students' ID, Name and his or her buying books");
				System.out.print("Your ID? ");
				ID = scn.next();
				System.out.print("Your Name? ");
				Name = scn.next();
				System.out.print("How many books? ");
				Books = scn.nextInt();
				sql = String.format("INSERT INTO classroom" +
								" VALUES('%s','%s','%d')", ID, Name, Books);
				stmt.executeUpdate(sql);
				ResultSet rs = stmt.executeQuery("SELECT * FROM classroom");
				System.out.println("StudentID\tStudentName\tBooks");
				while((rs.next())) { // Moves the cursor forward one row from its current position. 
					System.out.println(rs.getString("StudentID") + "\t" 
				+ rs.getString("StudentName") + "\t" + rs.getInt("Books"));
				}
			}
		}catch(Exception e) {
			e.printStackTrace();
		}
	}
}
