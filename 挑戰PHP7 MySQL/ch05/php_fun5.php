<?php
	$x = 2;
	function showDouble(&$x){
		$x = $x * 2 ;
		echo "函式中的值為：" . $x ."<br />";
	}
	showDouble($x);
	echo "函式外的值為：" . $x ."<br />";;
?>