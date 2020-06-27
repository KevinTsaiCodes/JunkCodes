<?php 
	include("connMysqlObj.php");
	if(isset($_POST["action"])&&($_POST["action"]=="delete")){	
		$sql_query = "DELETE FROM students WHERE cID=?";
		$stmt = $db_link -> prepare($sql_query);
		$stmt -> bind_param("i", $_POST["cID"]);
		$stmt -> execute();
		$stmt -> close();
		$db_link -> close();
		//重新導向回到主畫面
		header("Location: data.php");
	}
	$sql_select = "SELECT cID, cName, cSex, cBirthday, cEmail, cPhone, cAddr FROM students WHERE cID = ?";
	$stmt = $db_link -> prepare($sql_select);
	$stmt -> bind_param("i", $_GET["id"]);
	$stmt -> execute();
	$stmt -> bind_result($cid, $cname, $csex, $cbirthday, $cemail, $cphone, $caddr);
	$stmt -> fetch();
?>
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<title>學生資料管理系統</title>
</head>
<body>
<h1 align="center">學生資料管理系統 - 刪除資料</h1>
<p align="center"><a href="data.php">回主畫面</a></p>
<form action="" method="post" name="formDel" id="formDel">
  <table border="1" align="center" cellpadding="4">
    <tr>
      <th>欄位</th><th>資料</th>
    </tr>
    <tr>
      <td>姓名</td><td><?php echo $cname;?></td>
    </tr>
    <tr>
      <td>性別</td><td>
      <?php
      	if($csex=="M"){
      		echo "男";
      	}else{
      		echo "女";
      	}
   	  ?>
      </td>
    </tr>
    <tr>
      <td>生日</td><td><?php echo $cbirthday;?></td>
    </tr>
    <tr>
      <td>電子郵件</td><td><?php echo $cemail;?></td>
    </tr>
    <tr>
      <td>電話</td><td><?php echo $cphone;?></td>
    </tr>
    <tr>
      <td>住址</td><td><?php echo $caddr;?></td>
    </tr>
    <tr>
      <td colspan="2" align="center">
      <input name="cID" type="hidden" value="<?php echo $cid;?>">
      <input name="action" type="hidden" value="delete">
      <input type="submit" name="button" id="button" value="確定刪除這筆資料嗎？">
      </td>
    </tr>
  </table>
</form>
</body>
</html>
<?php 
	$stmt -> close();
	$db_link -> close();
?>