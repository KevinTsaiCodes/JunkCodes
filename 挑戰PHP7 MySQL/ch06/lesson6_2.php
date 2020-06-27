<?php
for ($i=2;$i<=9;$i++){
	for ($j=1;$j<=9;$j++){
		$calculate[$i][$j]=$i * $j;
	}
}

foreach ($calculate as $result => $resultData){
	foreach ($resultData as $detail => $detailData){
		echo $result." * ".$detail." = ".$detailData."<br />";
	}
}
?>