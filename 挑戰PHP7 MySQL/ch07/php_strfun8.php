<?php
$showStr = "南投縣 埔里鎮 愛蘭里 梅村路8段20號";
$showAddress = explode(" ",$showStr);
foreach ($showAddress as $value){
	echo $value."<br>";
}
?>