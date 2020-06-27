<?php
	$season = 5;
	if ($season >=1 && $season <= 3){
		echo "現在是 $season 月，是春季。";
	}elseif ($season>=4 && $season<= 5){
		echo "現在是 $season 月，是夏季。";
	}elseif ($season>=6 && $season<= 9){
		echo "現在是 $season 月，是秋季。";
	}elseif ($season>=10 && $season<=12){
		echo "現在是 $season 月，是冬季。";
	}else{
		echo "請設定 1~12 範圍的數值。";
	}
?>