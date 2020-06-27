<?php
$myString = "ABCDE";
for($i=0;$i<5;$i++){
$showWord = substr($myString,$i,1);
echo str_repeat($showWord, $i+1)."<br />";
}
?>