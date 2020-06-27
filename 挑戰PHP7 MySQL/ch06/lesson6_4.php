<?php
	$lottery = range(1,42,1);	
	$result = array_rand($lottery,6);
	sort($result);
	echo "本期樂透開獎號碼為：";
	foreach ($result as $val){
		echo $lottery[$val]." ";
	}
?>