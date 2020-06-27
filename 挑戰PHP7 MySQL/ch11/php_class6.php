<?php
class Student { 
	public $int_Id;		//座號
	public $str_Name;		//姓名
	public $str_Sex;		//性別
	public $int_Chinese;	//國文成績
	public $int_English;	//英語成績
	public $int_Maths;		//數學成績
	private $total_scores;	//總成績
	private $average_scores;	//平均分數 
	
	private function totalScores(){
		return $this->int_Chinese + $this->int_English + $this->int_Maths;
	}
	private function averageScores(){
		return round($this->total_scores/3);
	}
	public function setData($Id,$Name,$Sex,$Chinese,$English,$Maths){
		$this->int_Id = $Id;
		$this->str_Name = $Name;
		$this->str_Sex = $Sex;
		$this->int_Chinese = $Chinese;
		$this->int_English = $English;
		$this->int_Maths = $Maths;
		$this->total_scores=$this->totalScores();
		$this->average_scores=$this->averageScores();		
	}
	public function showData(){
		echo "座號：".$this->int_Id."<br />";
		echo "姓名：".$this->str_Name."<br />";
		echo "性別：".$this->str_Sex."<br />";
		echo "國文成績：".$this->int_Chinese."<br />";
		echo "英語成績：".$this->int_English."<br />";
		echo "數學成績：".$this->int_Maths."<br />";
		echo "總分：".$this->total_scores."<br />";
		echo "平均分數：".$this->average_scores."<br />";
	}
} 

$stdObject = new Student;
$stdObject->setData(1,"David","男",92,85,80);
$stdObject->showData();
?>