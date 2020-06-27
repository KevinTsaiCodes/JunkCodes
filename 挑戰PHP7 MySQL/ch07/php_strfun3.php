<?php
	$showStr="('文淵閣工作室'[自我介紹網址])http://www.e-happy.com.tw?aboutme=true";
	echo $showStr."<hr>";
	echo quotemeta($showStr)."<hr>";
	echo addslashes($showStr)."<hr>";
	echo addcslashes($showStr,"'[]")."<hr>";
	$showStr1 = addslashes($showStr);
	$showStr2 = addcslashes($showStr,"'[]");
	echo stripslashes($showStr1)."<hr>";
	echo stripcslashes($showStr2);
?>