<?php
	$path = __FILE__;
	echo pathinfo($path,PATHINFO_DIRNAME)."<br />";
	echo pathinfo($path,PATHINFO_BASENAME)."<br />";	
	echo pathinfo($path,PATHINFO_EXTENSION)."<br />";	
	echo pathinfo($path,PATHINFO_FILENAME)."<br />";
	$pathData = pathinfo($path);
	print_r($pathData);
?>