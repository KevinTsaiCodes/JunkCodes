<?php 
	header("Content-Type: text/html; charset=utf-8");
	include("connMysql.php");
	$seldb = @mysqli_select_db($db_link, "class");
	if (!$seldb) die("資料庫選擇失敗！");
	$sql_query = "SELECT * FROM students";
	$result = mysqli_query($db_link, $sql_query);
	$total_records = mysqli_num_rows($result);
?>
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<title>學生資料管理系統</title>
</head>
<body>
<h1 align="center">學生資料管理系統</h1>
<p align="center">目前資料筆數：<?php echo $total_records;?>，<a href="add.php">新增學生資料</a>。</p>
<table border="1" align="center">
  <!-- 表格表頭 -->
  <tr>
    <th>座號</th>
    <th>姓名</th>
    <th>性別</th>
    <th>生日</th>
    <th>電子郵件</th>
    <th>電話</th>
    <th>住址</th>
    <th>功能</th>
  </tr>
  <!-- 資料內容 -->
<?php
	while($row_result=mysqli_fetch_assoc($result)){
		echo "<tr>";
		echo "<td>".$row_result["cID"]."</td>";
		echo "<td>".$row_result["cName"]."</td>";
		echo "<td>".$row_result["cSex"]."</td>";
		echo "<td>".$row_result["cBirthday"]."</td>";
		echo "<td>".$row_result["cEmail"]."</td>";
		echo "<td>".$row_result["cPhone"]."</td>";
		echo "<td>".$row_result["cAddr"]."</td>";
		echo "<td><a href='update.php?id=".$row_result["cID"]."'>修改</a> ";
		echo "<a href='delete.php?id=".$row_result["cID"]."'>刪除</a></td>";
		echo "</tr>";
	}
?>
</table>
</body>
</html>