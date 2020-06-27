<?php
$showStr = "先生您好，歡迎光臨文淵閣工作室！";
echo $showStr. "<br>";
echo strtr($showStr, "文淵閣工作室", "文淵閣大飯店") . "<br>"; 
$replaceStr = array("先生"=>"小姐", "文淵閣工作室"=>"文淵閣大飯店");
echo strtr($showStr, $replaceStr) . "<br>"; 
echo str_replace("文淵閣工作室", "文淵閣", $showStr) . "<br>"; 
echo substr_replace($showStr, "小姐", 0, 6); 
?>