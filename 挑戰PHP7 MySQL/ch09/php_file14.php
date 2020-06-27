<?php
$filename = "php_file13.htm";
if(is_readable($filename)){
	readfile($filename);
}
?>