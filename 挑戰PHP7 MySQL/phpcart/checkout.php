<?php
require_once("connMysql.php");
//購物車開始
include("mycart.php");
session_start();
$cart =& $_SESSION['cart']; // 將購物車的值設定為 Session
if(!is_object($cart)) $cart = new myCart();
//購物車結束
//繫結產品目錄資料
$query_RecCategory = "SELECT category.categoryid, category.categoryname, category.categorysort, count(product.productid) as productNum FROM category LEFT JOIN product ON category.categoryid = product.categoryid GROUP BY category.categoryid, category.categoryname, category.categorysort ORDER BY category.categorysort ASC";
$RecCategory = $db_link->query($query_RecCategory);
//計算資料總筆數
$query_RecTotal = "SELECT count(productid) as totalNum FROM product";
$RecTotal = $db_link->query($query_RecTotal);
$row_RecTotal = $RecTotal->fetch_assoc();
?>
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<title>網路購物系統</title>
<link href="style.css" rel="stylesheet" type="text/css">
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
	return confirm('確定送出嗎？');
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
</head>
<body>
<table width="780" border="0" align="center" cellpadding="4" cellspacing="0" bgcolor="#FFFFFF">
  <tr>
    <td height="80" align="center" background="images/mlogo.png" class="tdbline"></td>
  </tr>
  <tr>
    <td class="tdbline"><table width="100%" border="0" cellspacing="0" cellpadding="10">
        <tr valign="top">
          <td width="200" class="tdrline"><div class="boxtl"></div>
            <div class="boxtr"></div>
            <div class="categorybox">
              <p class="heading"><img src="images/16-cube-orange.png" width="16" height="16" align="absmiddle"> 產品搜尋 <span class="smalltext">Search</span></p>
              <form name="form1" method="get" action="index.php">
                <p>
                  <input name="keyword" type="text" id="keyword" value="請輸入關鍵字" size="12" onClick="this.value='';">
                  <input type="submit" id="button" value="查詢">
                </p>
              </form>
              <p class="heading"><img src="images/16-cube-orange.png" width="16" height="16" align="absmiddle"> 價格區間 <span class="smalltext">Price</span></p>
              <form action="index.php" method="get" name="form2" id="form2">
                <p>
                  <input name="price1" type="text" id="price1" value="0" size="3">
                  -
                  <input name="price2" type="text" id="price2" value="0" size="3">
                  <input type="submit" id="button2" value="查詢">
                </p>
              </form>
            </div>
            <div class="boxbl"></div>
            <div class="boxbr"></div>
            <hr width="100%" size="1" />
            <div class="boxtl"></div>
            <div class="boxtr"></div>
            <div class="categorybox">
              <p class="heading"><img src="images/16-cube-orange.png" width="16" height="16" align="absmiddle"> 產品目錄 <span class="smalltext">Category</span></p>
              <ul>
                <li><a href="index.php?">所有產品 <span class="categorycount">(<?php echo $row_RecTotal["totalNum"];?>)</span></a></li>
                <?php	while($row_RecCategory = $RecCategory->fetch_assoc()){ ?>
                <li><a href="index.php?cid=<?php echo $row_RecCategory["categoryid"];?>"><?php echo $row_RecCategory["categoryname"];?> <span class="categorycount">(<?php echo $row_RecCategory["productNum"];?>)</span></a></li>
                <?php }?>
              </ul>
            </div>
            <div class="boxbl"></div>
            <div class="boxbr"></div></td>
          <td>
          <div class="subjectDiv"><span class="heading"><img src="images/16-cube-green.png" width="16" height="16" align="absmiddle"></span> 購物結帳</div>
            <div class="normalDiv">
              <?php if($cart->itemcount > 0) {?>
              <p class="heading"><img src="images/16-cube-orange.png" width="16" height="16" align="absmiddle"> 購物內容</p>
              <table width="90%" border="0" align="center" cellpadding="2" cellspacing="1">
                <tr>
                  <th bgcolor="#ECE1E1"><p>編號</p></th>
                  <th bgcolor="#ECE1E1"><p>產品名稱</p></th>
                  <th bgcolor="#ECE1E1"><p>數量</p></th>
                  <th bgcolor="#ECE1E1"><p>單價</p></th>
                  <th bgcolor="#ECE1E1"><p>小計</p></th>
                </tr>
                <?php		  
		  	$i=0;
			foreach($cart->get_contents() as $item) {
			$i++;
		  ?>
                <tr>
                  <td align="center" bgcolor="#F6F6F6" class="tdbline"><p><?php echo $i;?>.</p></td>
                  <td bgcolor="#F6F6F6" class="tdbline"><p><?php echo $item['info'];?></p></td>
                  <td align="center" bgcolor="#F6F6F6" class="tdbline"><p><?php echo $item['qty'];?></p></td>
                  <td align="center" bgcolor="#F6F6F6" class="tdbline"><p>$ <?php echo number_format($item['price']);?></p></td>
                  <td align="center" bgcolor="#F6F6F6" class="tdbline"><p>$ <?php echo number_format($item['subtotal']);?></p></td>
                </tr>
                <?php }?>
                <tr>
                  <td align="center" valign="baseline" bgcolor="#F6F6F6"><p>運費</p></td>
                  <td valign="baseline" bgcolor="#F6F6F6"><p>&nbsp;</p></td>
                  <td align="center" valign="baseline" bgcolor="#F6F6F6"><p>&nbsp;</p></td>
                  <td align="center" valign="baseline" bgcolor="#F6F6F6"><p>&nbsp;</p></td>
                  <td align="center" valign="baseline" bgcolor="#F6F6F6"><p>$ <?php echo number_format($cart->deliverfee);?></p></td>
                </tr>
                <tr>
                  <td align="center" valign="baseline" bgcolor="#F6F6F6"><p>總計</p></td>
                  <td valign="baseline" bgcolor="#F6F6F6"><p>&nbsp;</p></td>
                  <td align="center" valign="baseline" bgcolor="#F6F6F6"><p>&nbsp;</p></td>
                  <td align="center" valign="baseline" bgcolor="#F6F6F6"><p>&nbsp;</p></td>
                  <td align="center" valign="baseline" bgcolor="#F6F6F6"><p class="redword">$ <?php echo number_format($cart->grandtotal);?></p></td>
                </tr>
              </table>
              <hr width="100%" size="1" />
              <p class="heading"><img src="images/16-cube-orange.png" width="16" height="16" align="absmiddle"> 客戶資訊</p>
              <form action="cartreport.php" method="post" name="cartform" id="cartform" onSubmit="return checkForm();">
                <table width="90%" border="0" align="center" cellpadding="4" cellspacing="1">
                  <tr>
                    <th width="20%" bgcolor="#ECE1E1"><p>姓名</p></th>
                    <td bgcolor="#F6F6F6"><p>
                        <input type="text" name="customername" id="customername">
                        <font color="#FF0000">*</font></p></td>
                  </tr>
                  <tr>
                    <th width="20%" bgcolor="#ECE1E1"><p>電子郵件</p></th>
                    <td bgcolor="#F6F6F6"><p>
                        <input type="text" name="customeremail" id="customeremail">
                        <font color="#FF0000">*</font></p></td>
                  </tr>
                  <tr>
                    <th width="20%" bgcolor="#ECE1E1"><p>電話</p></th>
                    <td bgcolor="#F6F6F6"><p>
                        <input type="text" name="customerphone" id="customerphone">
                        <font color="#FF0000">*</font></p></td>
                  </tr>
                  <tr>
                    <th width="20%" bgcolor="#ECE1E1"><p>住址</p></th>
                    <td bgcolor="#F6F6F6"><p>
                        <input name="customeraddress" type="text" id="customeraddress" size="40">
                        <font color="#FF0000">*</font></p></td>
                  </tr>
                  <tr>
                    <th width="20%" bgcolor="#ECE1E1"><p>付款方式</p></th>
                    <td bgcolor="#F6F6F6"><p>
                        <select name="paytype" id="paytype">
                          <option value="ATM匯款" selected>ATM匯款</option>
                          <option value="線上刷卡">線上刷卡</option>
                          <option value="貨到付款">貨到付款</option>
                        </select>
                      </p></td>
                  </tr>
                  <tr>
                    <td colspan="2" bgcolor="#F6F6F6"><p><font color="#FF0000">*</font> 表示為必填的欄位</p></td>
                  </tr>
                </table>
                <hr width="100%" size="1" />
                <p align="center">
                  <input name="cartaction" type="hidden" id="cartaction" value="update">
                  <input type="submit" name="updatebtn" id="button3" value="送出訂購單">
                  <input type="button" name="backbtn" id="button4" value="回上一頁" onClick="window.history.back();">
                </p>
              </form>
            </div>
            <?php }else{ ?>
            <div class="infoDiv">目前購物車是空的。</div>
            <?php } ?></td>
        </tr>
      </table></td>
  </tr>
  <tr>
    <td height="30" align="center" background="images/album_r2_c1.jpg" class="trademark">© 2016 eHappy Studio All Rights Reserved.</td>
  </tr>
</table>
</body>
</html>
<?php $db_link->close();?>