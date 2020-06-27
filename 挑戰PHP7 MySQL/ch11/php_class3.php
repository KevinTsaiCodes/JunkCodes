<?php
class Student { 
	var $int_Id;		//座號
	var $str_Name;		//姓名
	var $str_Sex;		//性別
	var $int_Chinese;	//國文成績
	var $int_English;	//英語成績
	var $int_Maths;		//數學成績
	
	function __construct(){
		echo "*****學生資料開始*****<br />";
	}
	
	function setData($Id,$Name,$Sex,$Chinese,$English,$Maths){
		$this->int_Id = $Id;
		$this->str_Name = $Name;
		$this->str_Sex = $Sex;
		$this->int_Chinese = $Chinese;
		$this->int_English = $English;
		$this->int_Maths = $Maths;
	}
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
$stdObject1->setData(1,"David","男",92,85,80);
$stdObject1->showData();
$stdObject2 = new Student;
$stdObject2->setData(2,"Lily","女",87,90,76);
$stdObject2->showData();
?>