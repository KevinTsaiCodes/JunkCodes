// app10_4, タapp10_3岿粇
class CCircle              // ﹚竡摸CCircle
{
   private double pi=3.14;
   private double radius;

   public CCircle()              // ⊿Τま计篶じ
   {
   }
   public CCircle(double r)   // Τま计篶じ
   {
      radius=r;
   }
   public void setRadius(double r)
   {
      radius=r;
      System.out.println("radius="+radius);
   }
}

class CCoin extends CCircle  // ﹚竡CCoin摸膥┯CCircle摸
{
   private int value;

   public CCoin(double r, int v)    // CCoin()Τㄢま计篶じ
   {
      setRadius(r);        // 硓筁setRadius() methodㄓ砞﹚radiusΘ
      value=v;             // 砞﹚valueΘ
   }
}
public class app10_4
{
   public static void main(String args[])
   {
      CCoin coin1=new CCoin(2.5,10); // ミン㊣Τㄢま计篶じ
   }
}

/* app10_4 OUTPUT---
radius=2.5
------------------*/