// CBox.java, 此檔案置於pack7\subpack2資料夾內
package pack7.subpack2;    // 將CBox類別納入pack7.subpack2中
public class CBox
{
   private int length;
   private int width;
   private int height;

   public CBox(int l,int w,int h)
   {
      length=l;
      width=w;
      height=h;
   }
   public void show()
   {
      int vol=length*width*height;
      System.out.print("length="+length);
      System.out.print(", width="+width);
      System.out.print(", height="+height);
      System.out.println(", volume="+vol);
   }
}
