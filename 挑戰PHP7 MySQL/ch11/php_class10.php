<?php
class Student { 
	const title = "學生資料";	//類別常數
	public $int_Id;		//座號
	public $str_Name;		//姓名
	public $str_Sex;		//性別
	
	public function setData($Id,$Name,$Sex){
		$this->int_Id = $Id;
		$this->str_Name = $Name;
		$this->str_Sex = $Sex;
	}
	public function showData(){
		echo "座號：".$this->int_Id."<br />";
		echo "姓名：".$this->str_Name."<br />";
		echo "性別：".$this->str_Sex."<br />";
		echo "*****".self::title."結束*****<br/>";
	}
} 
echo "*****".Student::title."開始*****<br/>";
$stdObject = new Student();
$stdObject->setData(1,"David","男");
$stdObject->showData();
?>