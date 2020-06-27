<?php
class People{ 
	public $str_Name;			//姓名
	public $str_Sex;			//性別
	protected $int_Age;			//年齡
	protected $int_Weight;		//體重
	protected $int_Height;		//身高
	
	public function setData($Name,$Sex,$Age,$Height,$Weight){
		$this->str_Name = $Name;
		$this->str_Sex = $Sex;
		$this->int_Age = $Age;
		$this->int_Height = $Height;
		$this->int_Weight = $Weight;
	}
	//16歲以下兒童標準體重：標準體重 = 年齡 x 2 + 8
	public function calcWeight(){
		return $this->int_Age*2+8;
	}
}
?>