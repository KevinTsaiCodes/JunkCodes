<?php
	$testArray=array("img1","img2","img11");
	sort($testArray);
	print_r($testArray);
	echo "<br />";
	natsort($testArray);
	print_r($testArray);
?>