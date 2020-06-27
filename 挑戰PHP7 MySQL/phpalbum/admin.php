<?php
function GetSQLValueString($theValue, $theType) {
  switch ($theType) {
    case "string":
      $theValue = ($theValue != "") ? filter_var($theValue, FILTER_SANITIZE_MAGIC_QUOTES) : "";
      break;
    case "int":
      $theValue = ($theValue != "") ? filter_var($theValue, FILTER_SANITIZE_NUMBER_INT) : "";
      break;
    case "email":
      $theValue = ($theValue != "") ? filter_var($theValue, FILTER_VALIDATE_EMAIL) : "";
      break;
    case "url":
      $theValue = ($theValue != "") ? filter_var($theValue, FILTER_VALIDATE_URL) : "";
      break;      
  }
  return $theValue;
}
require_once("connMysql.php");
session_start();
//檢查是否經過登入
if(!isset($_SESSION["loginMember"]) || ($_SESSION["loginMember"]=="")){
	header("Location: login.php");
}
//執行登出動作
if(isset($_GET["logout"]) && ($_GET["logout"]=="true")){
	unset($_SESSION["loginMember"]);
	header("Location: index.php");
}
$sid = 0;
if(isset($_GET["id"])&&($_GET["id"]!="")){
	$sid = GetSQLValueString($_GET["id"],"int");
}
//刪除相簿
if(isset($_GET["action"])&&($_GET["action"]=="delete")){
	//刪除所屬相片
	$query_delphoto = "SELECT * FROM albumphoto WHERE album_id={$sid}";
	$delphoto = $db_link->query($query_delphoto);
	while($row_delphoto=$delphoto->fetch_assoc()){
		unlink("photos/".$row_delphoto["ap_picurl"]);
	}
	//刪除相簿
	$query_del1 = "DELETE FROM album WHERE album_id={$sid}";
	$query_del2 = "DELETE FROM albumphoto WHERE album_id={$sid}";
	$db_link->query($query_del1);
	$db_link->query($query_del2); 
	//重新導向回到主畫面
	header("Location: admin.php");
}
//預設每頁筆數
$pageRow_records = 8;
//預設頁數
$num_pages = 1;
//若已經有翻頁，將頁數更新
if (isset($_GET['page'])) {
  $num_pages = $_GET['page'];
}
//本頁開始記錄筆數 = (頁數-1)*每頁記錄筆數
$startRow_records = ($num_pages -1) * $pageRow_records;
//未加限制顯示筆數的SQL敘述句
$query_RecAlbum = "SELECT album.album_id , album.album_date , album.album_location , album.album_title , album.album_desc , albumphoto.ap_picurl, count( albumphoto.ap_id ) AS albumNum FROM album LEFT JOIN albumphoto ON album.album_id = albumphoto.album_id GROUP BY album.album_id , album.album_date , album.album_location , album.album_title , album.album_desc ORDER BY album_date DESC";
//加上限制顯示筆數的SQL敘述句，由本頁開始記錄筆數開始，每頁顯示預設筆數
$query_limit_RecAlbum = $query_RecAlbum." LIMIT {$startRow_records}, {$pageRow_records}";
//以加上限制顯示筆數的SQL敘述句查詢資料到 $RecAlbum 中
$RecAlbum = $db_link->query($query_limit_RecAlbum);
//以未加上限制顯示筆數的SQL敘述句查詢資料到 $all_RecAlbum 中
$all_RecAlbum = $db_link->query($query_RecAlbum);
//計算總筆數
$total_records = $all_RecAlbum->num_rows;
//計算總頁數=(總筆數/每頁筆數)後無條件進位。
$total_pages = ceil($total_records/$pageRow_records);
?>
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<title>網路相簿</title>
<link href="style.css" rel="stylesheet" type="text/css" />
<script language="javascript">
function deletesure(){
    if (confirm('\n您確定要刪除整個相簿嗎?\n刪除後無法恢復!\n')) return true;
    return false;
}
</script>
</head>
<body bgcolor="#cccccc">
<table width="778" border="0" align="center" cellpadding="0" cellspacing="0">
  <tr>
   <td height="124" valign="top" background="images/album_r1_c1.jpg"><div class="titleDiv"> [生活、旅行的記錄]<br />
   </div>
    <div class="menulink"><a href="admin.php">[管理首頁]</a> <a href="?logout=true">[登出系統]</a></div></td>
  </tr>
  <tr>
   <td background="images/album_r2_c1.jpg"><div id="mainRegion">
     <table width="90%" border="0" align="center" cellpadding="4" cellspacing="0">
       <tr>
         <td><div class="subjectDiv"> 網路相簿管理界面 </div>
           <div class="actionDiv">相簿總數: <?php echo $total_records;?>，<a href="adminadd.php">新增相簿</a></div>  
           <div class="normaldesc"></div>
           <?php while($row_RecAlbum=$RecAlbum->fetch_assoc()){ ?>
           <div class="albumDiv">
           <div class="picDiv"><a href="adminfix.php?id=<?php echo $row_RecAlbum["album_id"];?>"><?php if($row_RecAlbum["albumNum"]==0){?><img src="images/nopic.png" alt="暫無圖片" width="120" height="120" border="0" /><?php }else{?><img src="photos/<?php echo $row_RecAlbum["ap_picurl"];?>" alt="<?php echo $row_RecAlbum["album_title"];?>" width="120" height="120" border="0" /><?php }?></a></div>
           <div class="albuminfo"><a href="adminfix.php?id=<?php echo $row_RecAlbum["album_id"];?>"><?php echo $row_RecAlbum["album_title"];?></a><br />
             <span class="smalltext">共 <?php echo $row_RecAlbum["albumNum"];?> 張</span><br>
             <a href="?action=delete&id=<?php echo $row_RecAlbum["album_id"];?>" class="smalltext" onClick="javascript:return deletesure();">(刪除相簿)</a><br>
           </div>
           </div>
           <?php }?>
           <div class="navDiv">
		   <?php if ($num_pages > 1) { // 若不是第一頁則顯示 ?>
             <a href="?page=1">|&lt;</a> <a href="?page=<?php echo $num_pages-1;?>">&lt;&lt;</a>
           <?php }else{?>
             |&lt; &lt;&lt;  
		   <?php }?>
             <?php
		  	  for($i=1;$i<=$total_pages;$i++){
		  	  	  if($i==$num_pages){
		  	  	  	  echo $i." ";
		  	  	  }else{
		  	  	      echo "<a href=\"?page=$i\">$i</a> ";
		  	  	  }
		  	  }
		  	 ?>
             <?php if ($num_pages < $total_pages) { // 若不是最後一頁則顯示 ?>
             <a href="?page=<?php echo $num_pages+1;?>">&gt;&gt;</a> <a href="?page=<?php echo $total_pages;?>">&gt;|</a>
             <?php }else{?>
             &gt;&gt; &gt;|
			 <?php }?>
           </div></td>
         </tr>
     </table>
   </div></td>
  </tr>
  <tr>
   <td align="center" background="images/album_r2_c1.jpg" class="trademark">© 2016 eHappy Studio All Rights Reserved.</td>
  </tr>
  <tr>
   <td><img name="album_r4_c1" src="images/album_r4_c1.jpg" width="778" height="24" border="0" id="album_r4_c1" alt="" /></td>
  </tr>
</table>
</body>
</html>
<?php
	$db_link->close();
?>