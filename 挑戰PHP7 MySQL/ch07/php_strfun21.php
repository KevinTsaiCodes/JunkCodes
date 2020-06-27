<?php
$encodeStr = "abcd1234";
echo "原字串：" . $encodeStr . "<br>";
echo "crypt加密：" . crypt($encodeStr, substr($encodeStr, 0, 2)) . "<br>";
echo "md5加密：" . md5($encodeStr) . "<br>";
echo "sha1加密：" . sha1($encodeStr);
?>