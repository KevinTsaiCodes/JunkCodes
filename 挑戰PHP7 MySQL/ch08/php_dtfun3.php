<?php 
echo strftime("%A, %d %b %Y %H:%M:%S")."<br />";
echo gmstrftime("%A, %d %b %Y %H:%M:%S")."<br />";
setlocale(LC_ALL, '');
echo strftime("%A, %d %b %Y %H:%M:%S")."<br />";
echo gmstrftime("%A, %d %b %Y %H:%M:%S");
?>