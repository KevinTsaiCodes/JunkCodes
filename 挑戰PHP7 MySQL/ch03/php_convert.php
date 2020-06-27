<?php
$foo = 10;            // $foo 是一個整數
$str = "$foo";        // $str 是一個字符串
$fst = (string) $foo; // $fst 也是一個字符串

// 輸出 "they are the same"
if ($fst === $str) {
    echo "they are the same";
}
?>