<?php
$lines = file('php_file11.htm');
foreach ($lines as $line_num => $line) {
    echo "#<b>$line_num</b> : " . htmlspecialchars($line) . "<br />\n";
}
?> 