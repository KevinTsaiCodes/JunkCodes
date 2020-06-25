public class p02_NineTable {
	public static void main(String[] args) {
		for(int i=9;i>=1;i--) {
			for(int j=9;j>=1;j--) {
				if(i*j<10)
					System.out.print(i+"*"+j+"=0"+i*j+"\t");
				else
					System.out.print(i+"*"+j+"="+i*j+"\t");
			}
			System.out.println();
		}
	}
}
