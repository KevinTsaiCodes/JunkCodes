<?php
	function showMe(){
		static $msg;
		$msg += 2;		 
		echo $msg . "<br />";
	}
	showMe();
	showMe();
?>