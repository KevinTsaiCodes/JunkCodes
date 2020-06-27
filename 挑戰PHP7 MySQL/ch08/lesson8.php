<?php
//月曆表頭
$head = <<<calendar
  <tr>
    <td align="center">星期日</td>
    <td align="center">星期一</td>
    <td align="center">星期二</td>
    <td align="center">星期三</td>
    <td align="center">星期四</td>
    <td align="center">星期五</td>
    <td align="center">星期六</td>
  </tr>
calendar;
$timestamp = strtotime(date("Y-m-01"));
$thisMonthDays = date("t",$timestamp);		//該月有幾天
$thisMonthStart = date("w",$timestamp);		//該月1號是該週第幾天
$body = "<tr>";
for ($i=0;$i<$thisMonthStart;$i++){
	$body .= "<td align=\"center\">&nbsp;</td>";
}
$col = $thisMonthStart;
for ($i=1;$i<=$thisMonthDays;$i++){			//顯示日期
	if($col == 7){
		$body .= "</tr><tr>";
		$col = 0;
	}
	$body .= "<td align=\"center\">$i</td>";
	$col++;
}
for ($i=$col;$i<7;$i++){
	$body .= "<td align=\"center\">&nbsp;</td>";
}
$body .= "<tr>";
echo "<table width=\"100%\" border=\"1\">",$head,$body,"</table>";
?>