<?php
srand((double)microtime()*1000000);
$randval = rand(0,5);
switch($randval){
	case "0";
		echo "知足常樂。甘願做、歡喜受。";
		break;
	case "1";
		echo "生氣是拿別人的過錯來懲罰自己。";
		break;
	case "2";
		echo "人生多一份感恩，就多一份美化。";
		break;
	case "3";
		echo "縮小自我、擴大心胸，工作要歡喜，人與人要感恩。";
		break;
	case "4";
		echo "愛與感恩，能喜淨心中的煩惱。";
		break;
	case "5";
		echo "做好事要騰出時間，這是人生的目的，也是人生的義務。";
		break;		
}
?>