<?php
$fname = array("photo.jpg", "readme.doc","sheet.xls","note.txt");
$ftype = array(".txt"=>"文字檔", ".doc"=>"Word檔", ".xls"=>"Excel檔", ".jpg"=>"圖片檔");
foreach($fname as $val){
	$ft = strstr($val, ".");
	foreach($ftype as $name => $desc){
		if ($name == $ft){
			echo $val ." 是 " . $desc . "，附檔名是 ".substr($name,1)."。<br>";
			break;
		}
	}
}
?> 