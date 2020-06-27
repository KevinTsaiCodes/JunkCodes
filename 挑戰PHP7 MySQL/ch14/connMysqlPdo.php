<?php
	//資料庫主機設定
	$db_host = "localhost";
	$db_username = "root";
	$db_password = "1234";
	$db_name = "class";
	//錯誤處理
	try{
		//連線資料庫
		$db_link = new PDO("mysql:host={$db_host};dbname={$db_name};charset=utf8", $db_username, $db_password);
	} catch (PDOException $e) {
	    print "資料庫連結失敗，訊息:{$e->getMessage()}<br/>";
    	die();
    }
?>