<?php
$j = 0;$i = 1;
gstart:
  if ($i>10) goto gend;
  if ($i % 2 !=0){
    $j = $j + $i;
  }
  $i++;
  goto gstart;
gend:
  echo $j;
?>