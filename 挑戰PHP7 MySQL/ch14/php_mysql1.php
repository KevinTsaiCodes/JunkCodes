<?php 
	header("Content-Type: text/html; charset=utf-8");
	$db_link = mysqli_connect("localhost", "root", "1234");
	if (!$db_link) {
		echo "資料連結失敗";
	}else{
		echo "資料連結成功";
	}
?>