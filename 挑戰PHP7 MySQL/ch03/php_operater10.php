<?php
	$myIP = `ping www.e-happy.com.tw`;
	echo nl2br($myIP); //使用 nl2br() 函式將傳回值結果自動分行顯示。
?>
