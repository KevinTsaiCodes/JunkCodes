// app7_5, 簡單的範例
public class app7_5
{
   public static void main(String args[])
  {
     int score[]={9,14,6,18,2,10};  // 宣告一維陣列score
     largest(score);    // 將一維陣列score傳入largest() method
  }


  public static void largest(int arr[])
  {
     int max=arr[0];
     for(int i=0;i<arr.length;i++)
       if(max<arr[i])
          max=arr[i];
     System.out.println("largest num = "+max);
  }
}

/* app7_5 OUTPUT---
largest num = 18
-----------------*/