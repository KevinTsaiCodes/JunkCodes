<?php
$filename = fopen("test.txt","w+");	
if (flock($filename,LOCK_EX|LOCK_NB)){
  fwrite($filename,"這是新的檔案內容!!");
  flock($filename,LOCK_UN);
}else{
  echo "錯誤，檔案已遭其他鎖定使用。";
}
fclose($filename);
echo "檔案寫入成功，內容如下：<br />";
readfile("test.txt");
?>