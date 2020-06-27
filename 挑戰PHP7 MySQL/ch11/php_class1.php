<?php
class Student { 
	var $int_Id;		//座號
	var $str_Name;		//姓名
	var $str_Sex;		//性別
	var $int_Chinese;	//國文成績
	var $int_English;	//英語成績
	var $int_Maths;		//數學成績
	
	function showData(){
		echo "座號：".$this->int_Id."<br />";
		echo "姓名：".$this->str_Name."<br />";
		echo "性別：".$this->str_Sex."<br />";
		echo "國文成績：".$this->int_Chinese."<br />";
		echo "英語成績：".$this->int_English."<br />";
		echo "數學成績：".$this->int_Maths."<br />";
	}
} 

$stdObject1 = new Student();
$stdObject1->int_Id=1;
$stdObject1->str_Name="David";
$stdObject1->str_Sex="男";
$stdObject1->int_Chinese=92;
$stdObject1->int_English=85;
$stdObject1->int_Maths=80;
$stdObject1->showData();
?>