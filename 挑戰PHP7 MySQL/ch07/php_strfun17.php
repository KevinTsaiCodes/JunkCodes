<?php
$birthday = "1974-05-16";
sscanf($birthday, "%d-%d-%d", $by, $bm, $bd);
printf("我的生日西元%d年%d月%d日。", $by, $bm, $bd);
?>