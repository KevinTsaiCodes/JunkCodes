<?php  
	$mailData = "e-happy@e-happy.com.tw";
	if(preg_match ("/^([a-z0-9_\.-]+)@([\da-z\.-]+)\.([a-z\.]{2,6})$/",$mailData,$matchData)){
	     echo "郵件格式驗證成功，內容為：{$matchData[0]}。";
	  }else{
	     echo "郵件格式驗證失敗！";
	  } 
?>