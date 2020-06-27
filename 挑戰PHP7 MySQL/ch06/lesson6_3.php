<?php
	$lottery = range(1,42,1);
	shuffle($lottery);
	$result = array_slice($lottery,0,6);
	sort($result);
	echo "本期樂透開獎號碼為：";
	foreach ($result as $val){
		echo $val." ";
	}
?>