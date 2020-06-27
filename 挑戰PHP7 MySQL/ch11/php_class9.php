<?php
class Student { 
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
	}
	static function showMsg($msg){
		return $msg;
	}
} 
echo Student::showMsg("*****學生資料開始*****<br/>");
$stdObject1 = new Student();
$stdObject1->setData(1,"David","男");
$stdObject1->showData();
$stdObject2 = new Student();
$stdObject2->setData(2,"Lily","女");
$stdObject2->showData();
echo Student::showMsg("*****學生資料結束*****<br/>");
?>