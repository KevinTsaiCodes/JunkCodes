<?php
$dateStr = "2005-11-24 04:30:25"; 
$timestamp = strtotime($dateStr);
echo "出生時間：" . $dateStr . "<br />";
echo "懷孕日期：" . date("Y-m-d",strtotime("- 10 months",$timestamp)) . "<br />";
echo "滿月日期：" . date("Y-m-d",strtotime("+ 1 month",$timestamp)) . "<br />";
echo "週歲日期：" . date("Y-m-d",strtotime("+ 1 year",$timestamp)) . "<br />";
?>