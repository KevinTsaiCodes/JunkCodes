<?php
	printf("磁碟空間：%s bytes<br />剩餘空間：%s bytes。", number_format(disk_total_space(".")), number_format(disk_free_space(".")));
?>