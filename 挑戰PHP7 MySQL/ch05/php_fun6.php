<?php
	function showData(){
		$i = func_num_args();
		if ($i==0){
			echo "沒有指定學生資料！ <br/>";
		}else{
			echo "本班學生有：";
			$student = func_get_args();
			foreach ($student as $data){
				echo $data."，";
			}
			echo "最後一個為：".func_get_arg($i-1)."。<br />";
			
		}
	}
	showData();
	showData("李雲毓", "黃冠妮", "韋國書", "劉子芸", "李政昀");
?>