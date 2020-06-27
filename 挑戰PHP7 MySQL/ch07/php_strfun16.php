<?php
$birthday = "1974-05-16";
$barray = sscanf($birthday, "%d-%d-%d");
vprintf("我的生日西元%d年%d月%d日。", $barray);
?>