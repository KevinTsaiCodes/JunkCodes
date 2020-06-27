<?php
$filename = fopen("php_file13.htm","r");	
echo "<b>使用 fpassthru() 讀取檔案：</b><br />";
fpassthru($filename);
$filename = fopen("php_file13.htm","r");	
echo "<b>使用 stream_get_contents() 讀取檔案：</b><br />";
echo stream_get_contents($filename);
fclose($filename);
?>