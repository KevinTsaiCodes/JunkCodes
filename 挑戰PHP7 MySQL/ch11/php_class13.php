<?php
header("Content-Type: text/html; charset=utf-8");
require("php_class11.php");
class AdultCalc extends People{
	//男性標準體重 ＝（身高－80）× 0.7
	public function calcAdultWeight(){
			return round(($this->int_Height-80)*0.7);
	}
}
$myObject = new AdultCalc();
$myObject->setData("Perry","男",20,170,68);
echo $myObject->str_Name." 的標準體重為：";
echo $myObject->calcAdultWeight();
?>