<?php
$filename = fopen("php_file19.txt","r");
while($line = fgets($filename)){  
  echo $line;
  echo " [".ftell($filename)."]<br />";  
}
rewind($filename);
echo "<hr />";
while(!feof($filename)){
	for($i=0;$i<12;$i++){
		echo fgetc($filename);
	}
	echo " [".ftell($filename)."]<br />";
}
fclose($filename);
?>