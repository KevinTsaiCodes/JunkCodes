<?php 
	header("Content-Type: text/html; charset=utf-8");
	include("connMysqlObj.php");
	$sql_query = "SELECT * FROM students";
	$result = $db_link->query($sql_query);
	
	echo "全班同學人數為：".$result->num_rows;
	
	$db_link->close();
?>