<?php
	$fileDir = "C:\\XAMPP\PHP";
	$fileResource = opendir($fileDir);
	echo "<table border='1' width='100%'><tr><td width='20%' valign='top'>資料夾：</td><td>";
	while($fileList = readdir($fileResource)){
		if(is_dir($fileDir.'\\'.$fileList))	echo $fileList."<br />";
	}
	rewinddir($fileResource);
	echo "</td></tr><tr><td width='20%' valign='top'>檔案：</td><td>";
	while($fileList = readdir($fileResource)){
		if(is_file($fileDir.'\\'.$fileList))echo $fileList."<br />";
	}
	echo "</td></tr></table>";
	closedir($fileResource);
?>