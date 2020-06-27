<?php
	$fileDir = "C:\\XAMPP\PHP";
	$fileResource = scandir($fileDir);
	echo "<table border='1' width='100%'><tr><td width='20%' valign='top'>¸ê®Æ§¨¡G</td><td>";
	foreach($fileResource as $fileName){
		if(is_dir($fileDir.'\\'.$fileName))	echo $fileName."<br />";
	}
	echo "</td></tr><tr><td width='20%' valign='top'>ÀÉ®×¡G</td><td>";
	foreach($fileResource as $fileName){
		if(is_file($fileDir.'\\'.$fileName))	echo $fileName."<br />";
	}
	echo "</td></tr></table>";
?>