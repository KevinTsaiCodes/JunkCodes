<?php
$weekArray = array('星期日'=>'Sunday', '星期一'=>'Monday', '星期二'=>'Tuesday', '星期三'=>'Wednesday', '星期四'=>'Thursday', '星期五'=>'Friday', '星期六'=>'Saturday');

foreach($weekArray as $cweek => $eweek){
	echo $cweek . "的英文是" . $eweek ."<br />";
}
?>