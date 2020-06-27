<?php
require_once("connMysql.php");
//購物車開始
require_once("mycart.php");
session_start();
$cart =& $_SESSION['cart']; // 將購物車的值設定為 Session
if(!is_object($cart)) $cart = new myCart();

// 更新購物車內容
if(isset($_POST["cartaction"]) && ($_POST["cartaction"]=="update")){
	if(isset($_POST["updateid"])){
		$i=count($_POST["updateid"]);
		for($j=0;$j<$i;$j++){
			$cart->edit_item($_POST['updateid'][$j],$_POST['qty'][$j]);
		}
	}
	header("Location: mcart.php");
}
// 新增購物車內容
if(isset($_GET["cartaction"]) && ($_GET["cartaction"]=="add")){
	$cart->add_item($_GET['id'],$_GET['qty'],$_GET['price'],$_GET['name']);
	header("Location: mcart.php");
}
// 移除購物車內容
if(isset($_GET["cartaction"]) && ($_GET["cartaction"]=="remove")){
	$rid = intval($_GET['delid']);
	$cart->del_item($rid);
	header("Location: mcart.php");	
}
// 清空購物車內容
if(isset($_GET["cartaction"]) && ($_GET["cartaction"]=="empty")){
	$cart->empty_cart();
	header("Location: mcart.php");
}
?>
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PHP行動購物網</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link href="http://code.jquery.com/mobile/1.4.5/jquery.mobile-1.4.5.min.css" rel="stylesheet" type="text/css"/>
<style type="text/css">
.message {
	font-weight: bolder;
	color: #900;
	background-color: #FCC;
	text-align: center;
	padding: 20px;
	width: 60%;
	margin-right: auto;
	margin-left: auto;
	margin-top: 20px;
	margin-bottom: 20px;
}
.line_top {
	border-top-width: 1px;
	border-top-style: dotted;
	border-top-color: #666;
}
.datatable {
	border-collapse: collapse;
}
.datatable th {
	background-color: #ECE1E1;
}
.datatable tr {
	border-bottom: 1px solid #000;
}
.datatable tr.last {
	border-bottom: none;
}
</style>
<script language="javascript">
function checkForm(){	
	if(document.cartform.customername.value==""){
		alert("請填寫姓名!");
		document.cartform.customername.focus();
		return false;
	}
	if(document.cartform.customeremail.value==""){
		alert("請填寫電子郵件!");
		document.cartform.customeremail.focus();
		return false;
	}
	if(!checkmail(document.cartform.customeremail)){
		document.cartform.customeremail.focus();
		return false;
	}	
	if(document.cartform.customerphone.value==""){
		alert("請填寫電話!");
		document.cartform.customerphone.focus();
		return false;
	}
	if(document.cartform.customeraddress.value==""){
		alert("請填寫地址!");
		document.cartform.customeraddress.focus();
		return false;
	}
	if(confirm('確定送出嗎？')){
		cartform.submit();
	}
}
function checkmail(myEmail) {
	var filter  = /^([a-zA-Z0-9_\.\-])+\@(([a-zA-Z0-9\-])+\.)+([a-zA-Z0-9]{2,4})+$/;
	if(filter.test(myEmail.value)){
		return true;
	}
	alert("電子郵件格式不正確");
	return false;
}
</script>
<script src="http://code.jquery.com/jquery-1.11.1.min.js" type="text/javascript"></script>
<script src="http://code.jquery.com/mobile/1.4.5/jquery.mobile-1.4.5.min.js" type="text/javascript"></script>
</head>
<body>
<div data-role="page" id="page" data-add-back-btn="true">
  <div data-role="header" data-position="fixed">
    <h1><span class="subjectDiv">購物結帳</span></h1>
  </div>
  <div data-role="content">
    <?php if($cart->itemcount > 0) {?>
    <h2>購物內容 </h2>
    <table width="100%" border="0" cellpadding="2" cellspacing="0" class="datatable">
      <tr>
        <th width="20%">編號</th>
        <th>名稱</th>
        <th width="20%">數量</th>
        <th width="30%">小計</th>
      </tr>
      <?php
		$i=0;
		foreach($cart->get_contents() as $item) {
		$i++;
	  ?>
      <tr class="last">
        <td align="center"><?php echo $i;?>.</td>
        <td><?php echo $item['info'];?></td>
        <td align="center"><?php echo $item['qty'];?></td>
        <td align="right">$ <?php echo number_format($item['subtotal']);?></td>
      </tr>
      <?php }?>
      <tr>
        <td align="center">運費</td>
        <td>&nbsp;</td>
        <td align="right">&nbsp;</td>
        <td align="right">$ <?php echo number_format($cart->deliverfee);?></td>
      </tr>
      <tr>
        <td align="center">總計</td>
        <td>&nbsp;</td>
        <td align="right">&nbsp;</td>
        <td align="right">$ <?php echo number_format($cart->grandtotal);?></td>
      </tr>
    </table>
    <form action="mcartreport.php" method="post" name="cartform" id="cartform">
      <h2>聯絡資料</h2>
      <table width="100%" cellpadding="2" cellspacing="0" class="datatable">
        <tr>
          <th width="30%">姓名<font color="#FF0000">*</font></th>
          <td bgcolor="#F6F6F6"><input type="text" name="customername" id="customername"></td>
        </tr>
        <tr>
          <th width="20%">電子郵件<font color="#FF0000">*</font></th>
          <td bgcolor="#F6F6F6"><input type="text" name="customeremail" id="customeremail"></td>
        </tr>
        <tr>
          <th width="20%">電話<font color="#FF0000">*</font></th>
          <td bgcolor="#F6F6F6"><input type="text" name="customerphone" id="customerphone"></td>
        </tr>
        <tr>
          <th width="20%">住址<font color="#FF0000">*</font></th>
          <td bgcolor="#F6F6F6"><input name="customeraddress" type="text" id="customeraddress"></td>
        </tr>
        <tr>
          <th>付款方式<font color="#FF0000">*</font></th>
          <td><select name="paytype" id="paytype">
            <option value="ATM匯款" selected>ATM匯款</option>
            <option value="線上刷卡">線上刷卡</option>
            <option value="貨到付款">貨到付款</option>
          </select></td>
        </tr>
      </table>
      <input name="cartaction" type="hidden" id="cartaction" value="update">
    </form>
    <div data-role="controlgroup" data-type="horizontal" style="text-align:center">
     <a href="#" data-role="button" data-icon="back" data-rel="back">回上一頁</a>
     <a href="javascript:checkForm();" data-role="button" data-icon="forward">送出訂單</a>     
    </div>
    <?php }?>
  </div>
  <div data-role="footer" data-position="fixed">
    <div data-role="navbar">
      <ul>
        <li><a href="mindex.php" data-icon="grid">逛逛商店</a></li>
        <li><a href="mcart.php" data-icon="star" class="ui-btn-active" data-ajax="false">檢視購物車</a></li>
        <li><a href="#popupDialog" data-rel="popup" data-position-to="window" data-transition="pop" data-icon="gear">關於我們</a></li>
      </ul>
    </div>
  </div>
    <div data-role="popup" id="popupDialog" data-overlay-theme="b" data-theme="b" data-dismissible="false" style="max-width:400px;">
        <div data-role="header" data-theme="a">
        <h1>關於我們</h1>
        </div>
        <div role="main" class="ui-content">
            <h3 class="ui-title">行動購物網</h3>
            <p>期待能提供最新的產品，最優惠的價格，讓顧客能夠盡情的享受線上購物的樂趣，歡迎多多光臨！</p>        
            <a href="#" class="ui-btn ui-corner-all ui-shadow ui-btn-inline ui-btn-b" data-rel="back" data-transition="flow">關閉</a>
        </div>
    </div>     
    <div data-role="popup" id="popupDialog" data-overlay-theme="b" data-theme="b" data-dismissible="false" style="max-width:400px;">
        <div data-role="header" data-theme="a">
        <h1>關於我們</h1>
        </div>
        <div role="main" class="ui-content">
            <h3 class="ui-title">行動購物網</h3>
            <p>期待能提供最新的產品，最優惠的價格，讓顧客能夠盡情的享受線上購物的樂趣，歡迎多多光臨！</p>        
            <a href="#" class="ui-btn ui-corner-all ui-shadow ui-btn-inline ui-btn-b" data-rel="back" data-transition="flow">關閉</a>
        </div>
    </div>           
</div>
</body>
</html>
