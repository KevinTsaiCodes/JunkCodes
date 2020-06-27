<?php
$filename = fopen("php_file13.htm","r");	
while($line = fgets($filename)){
  echo $line;
}
fclose($filename);
?>