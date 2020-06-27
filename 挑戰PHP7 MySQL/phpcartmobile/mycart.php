<?php
include("wfCart/wfcart.php");
//擴充 wfCart 購物車運費
class myCart extends wfCart{
	var $deliverfee = 0;
	var $grandtotal = 0;
	
	function empty_cart()
	{ // empties / resets the cart
			$this->total = 0;
	        $this->itemcount = 0;
	        $this->deliverfee = 0;
	        $this->grandtotal = 0;
	        $this->items = array();
	        $this->itemprices = array();
	        $this->itemqtys = array();
	        $this->iteminfo = array();
	} // end of empty cart

	function _update_total()
	{ // internal function to update the total in the cart
	        $this->itemcount = 0;
		$this->total = 0;
                if(sizeof($this->items > 0))
		{
                        foreach($this->items as $item) {
                                $this->total = $this->total + ($this->itemprices[$item] * $this->itemqtys[$item]);
				$this->itemcount++;
			}
		}
		if($this->total >= 50000){
			$this->deliverfee = 0;			
		}else{
			$this->deliverfee = 500;						
		}
		$this->grandtotal = $this->total+$this->deliverfee;		
	}
}
?>
