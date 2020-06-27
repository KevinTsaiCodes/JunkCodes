<?php 
require_once("connMysql.php");
if(isset($_POST["customername"]) && ($_POST["customername"]!="")){
	//購物車開始
	require_once("mycart.php");
	session_start();
	$cart =& $_SESSION['cart']; // 將購物車的值設定為 Session
	if(!is_object($cart)) $cart = new myCart();
	//購物車結束	
	//新增訂單資料
	$sql_query = "INSERT INTO orders (total ,deliverfee ,grandtotal ,customername ,customeremail ,customeraddress ,customerphone ,paytype) VALUES (?, ?, ?, ?, ?, ?, ?, ?)";
	$stmt = $db_link->prepare($sql_query);
	$stmt->bind_param("iiisssss", $cart->total, $cart->deliverfee, $cart->grandtotal, $_POST["customername"], $_POST["customeremail"], $_POST["customeraddress"], $_POST["customerphone"], $_POST["paytype"]);
	$stmt->execute();
	//取得新增的訂單編號
	$o_pid = $stmt->insert_id;
	$stmt->close();
	//新增訂單內貨品資料
	if($cart->itemcount > 0) {
		foreach($cart->get_contents() as $item) {
			$sql_query="INSERT INTO orderdetail (orderid ,productid ,productname ,unitprice ,quantity) VALUES (?, ?, ?, ?, ?)";
			$stmt = $db_link->prepare($sql_query);
			$stmt->bind_param("iisii", $o_pid, $item['id'], $item['info'], $item['price'], $item['qty']);
			$stmt->execute();
			$stmt->close();
		}
	}
	//郵寄通知
	$cname = $_POST["customername"];
	$cmail = $_POST["customeremail"];
	$ctel = $_POST["customerphone"];
	$caddress = $_POST["customeraddress"];
	$cpaytype = $_POST["paytype"];
	$total = $cart->grandtotal;
	$mailcontent=<<<msg
	親愛的 $cname 您好：
	感謝您的光臨
	本次消費詳細資料如下：
	--------------------------------------------------
	訂單編號： $o_pid 
	客戶姓名：$cname 
	電子郵件： $cmail 
	電話： $ctel 
	住址： $caddress 
	付款方式： $cpaytype 
	消費金額：	$total 
	--------------------------------------------------
	希望能再次為您服務 
	
	網路購物公司 敬上
msg;
	$mailFrom="=?UTF-8?B?" . base64_encode("網路購物系統") . "?= <service@e-happy.com.tw>";
	$mailto = $_POST["customeremail"];
	$mailSubject="=?UTF-8?B?" . base64_encode("網路購物系統訂單通知"). "?=";
	$mailHeader="From:".$mailFrom."\r\n";
	$mailHeader.="Content-type:text/html;charset=UTF-8";
	if(!@mail($mailto,$mailSubject,nl2br($mailcontent),$mailHeader)) die("郵寄失敗！");
	//清空購物車
	$cart->empty_cart();
}	
?>
<script language="javascript">
alert("感謝您的購買，我們將儘快進行處理。");
window.location.href="mindex.php";
</script>