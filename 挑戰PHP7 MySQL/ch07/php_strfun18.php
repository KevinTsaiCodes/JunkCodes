<?php
$queryStr = "name=perry&age=3&height=95&weight=14";
parse_str($queryStr);
printf ("大家好，我是%s ，我今年%d歲，身高 %d公分，體重%d公斤。", $name, $age, $height, $weight);
?>