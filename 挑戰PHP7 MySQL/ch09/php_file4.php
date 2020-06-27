<?php
	$path = __FILE__;
	clearstatcache();
	$pathData = stat($path);
	print_r($pathData);
?>