<?php 
	header("Content-Type: text/html; charset=utf-8");
	include("connMysql.inc");
	$seldb = @mysqli_select_db($db_link, "class");
	if (!$seldb) die("資料庫選擇失敗！");
	$sql_query = "SELECT * FROM students";
	$result = mysqli_query($db_link, $sql_query);
	mysqli_data_seek($result,4);
	$row_result=mysqli_fetch_assoc($result);
	foreach($row_result as $item=>$value){
		echo $item."=".$value."<br />";
	}
?>