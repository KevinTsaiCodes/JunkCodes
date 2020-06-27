<?php 
	header("Content-Type: text/html; charset=utf-8");
	include("connMysqlPdo.php");
	
	$sex = "M";	$id = 5;
	$stmt = $db_link->prepare("SELECT * FROM students WHERE cSex = ? AND cID <= ?");
	if($stmt->execute(array($sex, $id))){
		while($row=$stmt->fetch()){
			echo "座號：{$row['cID']}<br>姓名：{$row['cName']}<br>電子郵件：{$row['cEmail']}<br>電話：{$row['cPhone']}<hr>";
		}
	}
?>