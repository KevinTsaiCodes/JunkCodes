<?php
function cmp($a, $b)
{
    if ($a == $b) {
        return 0;
    }
    return ($a < $b) ? -1 : 1;
}
$aNum = array(3, 2, 5, 6, 1);
usort($aNum, "cmp");
foreach ($aNum as $key => $value) {
    echo "$key: $value <br />";
}
?>