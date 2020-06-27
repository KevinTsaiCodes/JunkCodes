<?php
	$path = __FILE__;
	clearstatcache();
	echo fileinode($path)."<br />";
	echo fileperms($path)."<br />";
	echo fileowner($path)."<br />";
	echo filegroup($path)."<br />";
	echo filesize($path)."<br />";
	echo fileatime($path)."<br />";
	echo filemtime($path)."<br />";
	echo filectime($path)."<br />";
	echo filetype($path)."<br />";
?>