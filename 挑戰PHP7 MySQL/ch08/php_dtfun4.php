<?php
$dateStr = "2005-11-24 04:30:25"; 
sscanf($dateStr, "%d-%d-%d %d:%d:%d", $y, $m, $d, $h, $i, $s);
$timestamp1 = mktime($h,$i,$s,$m,$d,$y);
echo $timestamp1."<br />";
$timestamp2 = strtotime($dateStr);
echo $timestamp2;
?>