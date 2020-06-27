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
//新增相簿
if(isset($_POST["action"])&&($_POST["action"]=="add")){
	$query_insert = "INSERT INTO album (album_title, album_date, album_location, album_desc) VALUES (?, ?, ?, ?)";
	$stmt = $db_link->prepare($query_insert);
	$stmt->bind_param("ssss", 
	 GetSQLValueString($_POST["album_title"], "string"),
	 GetSQLValueString($_POST["album_date"], "string"),
	 GetSQLValueString($_POST["album_location"], "string"),
	 GetSQLValueString($_POST["album_desc"], "string"));
	$stmt->execute();

	//取得新增的相簿編號
	$album_pid = $stmt->insert_id;
	$stmt->close();
	
	for ($i=0; $i<count($_FILES["ap_picurl"]["name"]); $i++) {
	  if ($_FILES["ap_picurl"]["tmp_name"][$i] != "") {
		  $query_insert = "INSERT INTO albumphoto (album_id, ap_date, ap_picurl, ap_subject) VALUES (?, NOW(), ?, ?)";
		  $stmt = $db_link->prepare($query_insert);
		  $stmt->bind_param("iss", 
		   GetSQLValueString($album_pid, "int"),
		   GetSQLValueString($_FILES["ap_picurl"]["name"][$i], "string"),
		   GetSQLValueString($_POST["ap_subject"][$i], "string"));
		  $stmt->execute();
		  if(!move_uploaded_file($_FILES["ap_picurl"]["tmp_name"][$i] , "photos/" . $_FILES["ap_picurl"]["name"][$i])) die("檔案上傳失敗！");
		  $stmt->close();
	  }
	}
	
	//重新導向到修改畫面
	header("Location: adminfix.php?id={$album_pid}");
}	
?>
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<title>網路相簿</title>
<link href="style.css" rel="stylesheet" type="text/css" />
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
            <td><div class="subjectDiv"> 網路相簿管理界面-相增相簿</div>
              <div class="actionDiv"><a href="#" onClick="window.history.back();">回上一頁</a></div>
              <div class="normalDiv">
                <form action="" method="post" enctype="multipart/form-data" name="form1" id="form1">
                  <p>相簿名稱：<input type="text" name="album_title" id="album_title" /></p>
                  <p>拍攝時間：<input name="album_date" type="text" id="album_date" value="<?php echo date("Y-m-d H:i:s");?>" /></p>
    			  <p>拍攝地點 ：<input type="text" name="album_location" id="album_location" /></p>
                  <p>相簿說明：<textarea name="album_desc" id="album_desc" cols="45" rows="5"></textarea></p>
                  <hr />
                  <p>照片1<input type="file" name="ap_picurl[]" id="ap_picurl[]" />
                  說明1：<input type="text" name="ap_subject[]" id="ap_subject[]" /></p>
                  <p>照片2<input type="file" name="ap_picurl[]" id="ap_picurl[]" />
                  說明2：<input type="text" name="ap_subject[]" id="ap_subject[]" /></p>
                  <p>照片3<input type="file" name="ap_picurl[]" id="ap_picurl[]" />
                  說明3：<input type="text" name="ap_subject[]" id="ap_subject[]" /></p>
                  <p>照片4<input type="file" name="ap_picurl[]" id="ap_picurl[]" />
                  說明4：<input type="text" name="ap_subject[]" id="ap_subject[]" /></p>
                  <p>照片5<input type="file" name="ap_picurl[]" id="ap_picurl[]" />
                  說明5：<input type="text" name="ap_subject[]" id="ap_subject[]" /></p>
                  <p>
                    <input name="action" type="hidden" id="action" value="add">    
                    <input type="submit" name="button" id="button" value="確定新增" />
                    <input type="button" name="button2" id="button2" value="回上一頁" onClick="window.history.back();" />
                  </p>
                </form>
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