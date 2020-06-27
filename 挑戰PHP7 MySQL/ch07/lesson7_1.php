<?php
$myJoke = <<<joke
A Window Seat\nA: Which do you prefer, a window seat or an aisle seat?\nB: I always prefer a window seat.\nA: Why?\nB: In case some bad thing happen I can jump out from it.
joke;
echo ucwords(nl2br($myJoke));
?>