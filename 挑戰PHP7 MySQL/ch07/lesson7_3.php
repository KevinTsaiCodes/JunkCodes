<?php
$student = array(array('姓名'=>'李雲毓', '生日'=>'2000/3/14', '電話'=>'0227042762'), 
				array('姓名'=>'黃冠妮', '生日'=>'2000/6/6', '電話'=>'0220938123'),
				array('姓名'=>'韋國書', '生日'=>'2000/7/15', '電話'=>'0225021314'),
				array('姓名'=>'劉子芸', '生日'=>'2000/8/7', '電話'=>'0425307996'),
				array('姓名'=>'李政昀', '生日'=>'2000/12/24', '電話'=>'0227408965'));

foreach($student as $data){
	$birth = sscanf($data['生日'], "%d/%d/%d");
	$cbirth = vsprintf("%d年%02d月%02d日", $birth);
	$phone = sprintf("(%02d)%08d",substr($data['電話'],0,2),substr($data['電話'],2,10));
	echo "姓名：".$data['姓名']."、生日：".$cbirth."、電話：".$phone."。<br />";
}
?>