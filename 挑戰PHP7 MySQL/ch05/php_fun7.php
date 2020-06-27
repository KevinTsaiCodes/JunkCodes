<?php
	$msg = "這是全域變數<br />";
	function showMsg(){
		$msg = "這是區域變數<br />";		 
		echo $msg;
	}
	echo $msg;
	showMsg();
	echo $msg;
?>