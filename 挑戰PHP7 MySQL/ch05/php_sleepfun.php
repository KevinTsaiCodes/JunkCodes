<?php
// 顯示目前系統時間的時分秒
echo date('h:i:s') . "<br />";
// 延遲程式10秒後再執行
sleep(10);
// 再顯示一次
echo date('h:i:s');
?>
