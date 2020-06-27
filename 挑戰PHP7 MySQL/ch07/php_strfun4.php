<?php
	$showStr1="大家";
	$showStr2="  一起學  ";
	$showStr3="PHP！";
	echo $showStr1.$showStr2.$showStr3."<br>";
	echo $showStr1.trim($showStr2).$showStr3."<br>";
	echo $showStr1.ltrim($showStr2).$showStr3."<br>";
	echo $showStr1.rtrim($showStr2).$showStr3."<br>";
	echo $showStr1.chop($showStr2).$showStr3;
?>