<?php 
include("connMysql.php");
if (!@mysqli_select_db($db_link, "class")) die("資料庫選擇失敗！");
if(isset($_POST["action"])&&($_POST["action"]=="update")){	
	$sql_query = "UPDATE students SET ";
	$sql_query .= "cName='".$_POST["cName"]."',";
	$sql_query .= "cSex='".$_POST["cSex"]."',";
	$sql_query .= "cBirthday='".$_POST["cBirthday"]."',";
	$sql_query .= "cEmail='".$_POST["cEmail"]."',";
	$sql_query .= "cPhone='".$_POST["cPhone"]."',";
	$sql_query .= "cAddr='".$_POST["cAddr"]."' ";
	$sql_query .= "WHERE cID=".$_POST["cID"];	
	mysqli_query($db_link, $sql_query);
	//重新導向回到主畫面
	header("Location: data.php");
}
$sql_db = "SELECT * FROM students WHERE cID=".$_GET["id"];
$result = mysqli_query($db_link, $sql_db);
$row_result=mysqli_fetch_assoc($result);
?>
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<title>學生資料管理系統</title>
</head>
<body>
<h1 align="center">學生資料管理系統 - 修改資料</h1>
<p align="center"><a href="data.php">回主畫面</a></p>
<form action="" method="post" name="formFix" id="formFix">
  <table border="1" align="center" cellpadding="4">
    <tr>
      <th>欄位</th><th>資料</th>
    </tr>
    <tr>
      <td>姓名</td><td><input type="text" name="cName" id="cName" value="<?php echo $row_result["cName"];?>"></td>
    </tr>
    <tr>
      <td>性別</td><td>
      <input type="radio" name="cSex" id="radio" value="M" <?php if($row_result["cSex"]=="M") echo "checked";?>>男
      <input type="radio" name="cSex" id="radio" value="F" <?php if($row_result["cSex"]=="F") echo "checked";?>>女
      </td>
    </tr>
    <tr>
      <td>生日</td><td><input type="text" name="cBirthday" id="cBirthday" value="<?php echo $row_result["cBirthday"];?>"></td>
    </tr>
    <tr>
      <td>電子郵件</td><td><input type="text" name="cEmail" id="cEmail" value="<?php echo $row_result["cEmail"];?>"></td>
    </tr>
    <tr>
      <td>電話</td><td><input type="text" name="cPhone" id="cPhone" value="<?php echo $row_result["cPhone"];?>"></td>
    </tr>
    <tr>
      <td>住址</td><td><input name="cAddr" type="text" id="cAddr" size="40" value="<?php echo $row_result["cAddr"];?>"></td>
    </tr>
    <tr>
      <td colspan="2" align="center">
      <input name="cID" type="hidden" value="<?php echo $row_result["cID"];?>">
      <input name="action" type="hidden" value="update">
      <input type="submit" name="button" id="button" value="更新資料">
      <input type="reset" name="button2" id="button2" value="重新填寫">
      </td>
    </tr>
  </table>
</form>
</body>
</html>