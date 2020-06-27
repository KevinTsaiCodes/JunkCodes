<?php 
	header("Content-Type: text/html; charset=utf-8");
	include("connMysqlObj.php");
	$sql_query = "SELECT * FROM students";
	$result = $db_link->query($sql_query);
	$result->data_seek(4);
	
	$row_result=$result->fetch_assoc();
	foreach($row_result as $item=>$value){
		echo $item."=".$value."<br />";
	}
	
	$db_link->close();
?>