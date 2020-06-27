<?php 
$a = array(20,25,18,34,20,45);
$b = array('礦泉水','泡麵','洋芋片','餅乾','八寶粥','罐頭');
array_multisort($a,SORT_DESC,$b,SORT_ASC);
for($i=0;$i<count($a);$i++){
	echo $b[$i].' 售價 $'.$a[$i].'<br />';
}
?>