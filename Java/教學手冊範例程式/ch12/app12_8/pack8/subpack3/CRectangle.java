// CRectangle.java, 此檔案置於pack8\subpack3資料夾內
package pack8.subpack3; // 將CRectangle類別納入pack8.subpack3中
public class CRectangle
{
   private int length;
   private int width;  
   
   public CRectangle(int l,int w)
   {
      length=l;
      width=w;     
   }
   public void show()
   {     
      System.out.print("length="+length);
      System.out.print(", width="+width);
      System.out.println(", area="+length*width);
   }
}
