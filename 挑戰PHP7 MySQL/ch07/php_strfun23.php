<?php
if(isset($_POST["passwd"]) && $_POST["passwd"]!=""){
	$passStr = md5("abcd1234");	
	$inputStr = $_POST["passwd"];
	if(md5($inputStr)==$passStr){
		echo "密碼 $inputStr 驗證通過<br>";
		echo "<a href='#' onclick='window.history.back();';>回上一頁</a>";
	}else{
		echo "密碼 $inputStr 驗證失敗<br>";	
		echo "<a href='#' onclick='window.history.back();';>回上一頁</a>";
	}
}else{
?>
<form action="" name="form1" method="POST">
密碼
<input name="passwd" type="password" id="passwd">
<input type="submit" name="Submit" value="驗證">
</form>
<?php }?>
