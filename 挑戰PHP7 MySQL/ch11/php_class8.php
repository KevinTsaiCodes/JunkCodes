<?php
class Student { 
	public $int_Id;		//座號
	public $str_Name;		//姓名
	static public $countNum=0;		//靜態成員記錄物件數
	
	function __construct(){
		Student::$countNum++;
	}
	function __destruct(){
		Student::$countNum--;
	}
	public function setData($Id,$Name){
		$this->int_Id = $Id;
		$this->str_Name = $Name;	
	}
} 

$stdObject1 = new Student();
$stdObject1->setData(1,"David","男");
$stdObject2 = new Student;
$stdObject2->setData(2,"Lily","女");
$stdObject3 = new Student;
$stdObject3->setData(3,"Perry","男");
$stdObject3 = NULL;
echo "目前的學生物件有".Student::$countNum."個。";
?>