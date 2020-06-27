<?php
$myLanguage = "PHP"; 
//nowdoc語法結構表示字串 
$showStr = <<<'Msg'
	我最喜歡的網頁程式語言是： $myLanguage <br />
	許多學生都說："It's easy, It's good."
Msg;
echo $showStr;
?>