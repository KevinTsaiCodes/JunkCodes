<?php
	$showStr="你學<strong>PHP</strong><br>我學<strong>PHP</strong><br>大家來學<strong>PHP</strong>";
	echo $showStr."<hr>";
	echo strip_tags($showStr,'<br>');
?>