<?php
	$i=count($_FILES["fileUpload"]["name"]);
	for ($j=0;$j<$i;$j++){
		if($_FILES["fileUpload"]["error"][$j]==0){
			if(move_uploaded_file($_FILES["fileUpload"]["tmp_name"][$j], "./".$_FILES["fileUpload"]["name"][$j])){
				echo $_FILES["fileUpload"]["name"][$j]."上傳成功!<br />";
			}else{
				echo $_FILES["fileUpload"]["name"][$j]."上傳失敗!<br />";
			}
		}
	}
?>