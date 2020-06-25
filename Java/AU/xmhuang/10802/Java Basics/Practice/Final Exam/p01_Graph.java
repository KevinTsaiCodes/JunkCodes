public class p01_Graph {
	
	public static void main(String[] args) {
		for(int i=1;i<=15;i++) {
			for(int j=1;j<=15;j++) {
				if(j==8 || j==9) {
					System.out.print(" ");
					continue;
				}
				System.out.print("*");
			}
			if(i==7) {
				System.out.println("\n");
				continue;
			}
			System.out.println();
		}
	}
}
