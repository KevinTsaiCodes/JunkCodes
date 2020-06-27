<?php
echo "getdate() 函式的使用<hr />";
$nowTime = getdate();
foreach($nowTime as $Key => $Value){
	echo "$Key => $Value <br />";
}
echo "<hr />localtime() 函式的使用<hr />";
$nowTime = localtime(time(),1);
foreach($nowTime as $Key => $Value){
	echo "$Key => $Value <br />";
}
echo "<hr />gettimeofday() 函式的使用<hr />";
$nowTime = gettimeofday();
foreach($nowTime as $Key => $Value){
	echo "$Key => $Value <br />";
}
?>
