<?php 
	header("Content-Type: text/html; charset=utf-8");
	include("connMysqlObj.php");

	if ($result = $db_link->query("SELECT DATABASE()")) {
	    $row = $result->fetch_row();
	    printf("目前連接的資料庫是「%s」<br>", $row[0]);
	    $result->close();
	}
	/* change db to world db */
	$db_link->select_db("test");

	/* return name of current default database */
	if ($result = $db_link->query("SELECT DATABASE()")) {
	    $row = $result->fetch_row();
	    printf("目前連接的資料庫是「%s」<br>", $row[0]);
	    $result->close();
	}
		
	$db_link->close();	
?>