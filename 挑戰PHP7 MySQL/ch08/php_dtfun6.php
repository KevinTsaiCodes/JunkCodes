<?php
$thisYear = date("Y");
if(checkdate(2, 29, $thisYear)){
	echo "今年 $thisYear 年是閏年";
}else{
	echo "今年 $thisYear 年不是閏年";
}
?>