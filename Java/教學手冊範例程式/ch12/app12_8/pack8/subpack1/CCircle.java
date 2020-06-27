// CCircle.java, 此檔案置於pack8\subpack1資料夾內
package pack8.subpack1; // 將CCircle類別納入pack8.subpack1中
public class CCircle
{
   final static double PI=3.14;
   private double radius;

   public CCircle(double r)
   {
      radius=r;
   }
   public void show()
   {
      System.out.print("radius="+radius);
      System.out.println(", area="+PI*radius*radius);
   }
}
