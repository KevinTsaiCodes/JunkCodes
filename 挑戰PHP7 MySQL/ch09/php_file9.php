<?php
	if($_FILES["fileUpload"]["error"]==0){
		if(move_uploaded_file($_FILES["fileUpload"]["tmp_name"], "./".$_FILES["fileUpload"]["name"])){
			echo "上傳成功<br />";
			echo "檔案名稱：".$_FILES["fileUpload"]["name"]."<br />";
			echo "檔案類型：".$_FILES["fileUpload"]["type"]."<br />";
			echo "檔案大小：".$_FILES["fileUpload"]["size"]."<br />";
		}else{
			echo "上傳失敗! ";
			echo "<a href='javascript:window.history.back();'>回上一頁</a>";
		}
	}
?>