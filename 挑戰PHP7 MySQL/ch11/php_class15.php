<?php
header("Content-Type: text/html; charset=utf-8");
require("php_class11.php");
class AdultCalc extends People{	
	public function calcWeight(){
		if($this->int_Age<=16){
			return parent::calcWeight();
		}else{
			if($this->str_Sex=="男"){
				//男性標準體重 ＝（身高－80）× 0.7
				return round(($this->int_Height-80)*0.7);
			}else{
				//女性標準體重 ＝（身高－70）× 0.6
				return round(($this->int_Height-70)*0.6 );
			}
		}
	}
}
$boyObject = new AdultCalc();
$boyObject->setData("David","男",24,181,89);
echo $boyObject->str_Name." 的標準體重為：";
echo $boyObject->calcWeight();
echo "<br />";
$girlObject = new AdultCalc();
$girlObject->setData("Lily","女",22,162,51);
echo $girlObject->str_Name." 的標準體重為：";
echo $girlObject->calcWeight();
echo "<br />";
$childObject = new AdultCalc();
$childObject->setData("Perry","男",14,158,40);
echo $childObject->str_Name." 的標準體重為：";
echo $childObject->calcWeight();
?>