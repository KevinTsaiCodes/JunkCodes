<?php
$strURL = "公司網站 http://www.e-happy.com.tw";
$regex = "/(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?/";
echo preg_replace($regex,"<a href=\"\\0\" target=\"_blank\">\\0</a>",$strURL);
?>
