<?php
$filename = fopen("php_file18.gif","rb");	
$contents = fread($filename,filesize("php_file18.gif"));
fclose($filename);
header('Content-Type: image/gif');
echo $contents;
?>