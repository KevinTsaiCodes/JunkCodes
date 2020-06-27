<?php  
	$strData = "公司信箱 service@e-happy.com.tw，我的信箱是 david@e-happy.com.tw";
	$regex = "/([a-z0-9_\.-]+)@([\da-z\.-]+)\.([a-z\.]{2,6})/";
	echo preg_replace($regex,"<a href=\"mailto:\\0\">\\0</a>",$strData);
?>