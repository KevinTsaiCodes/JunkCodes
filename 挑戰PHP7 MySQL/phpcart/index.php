<?php
require_once("connMysql.php");
//預設每頁筆數
$pageRow_records = 6;
//預設頁數
$num_pages = 1;
//若已經有翻頁，將頁數更新
if (isset($_GET['page'])) {
  $num_pages = $_GET['page'];
}
//本頁開始記錄筆數 = (頁數-1)*每頁記錄筆數
$startRow_records = ($num_pages -1) * $pageRow_records;
//若有分類關鍵字時未加限制顯示筆數的SQL敘述句
if(isset($_GET["cid"])&&($_GET["cid"]!="")){
	$query_RecProduct = "SELECT * FROM product WHERE categoryid=? ORDER BY productid DESC";
	$stmt = $db_link->prepare($query_RecProduct);
	$stmt->bind_param("i", $_GET["cid"]);
//若有搜尋關鍵字時未加限制顯示筆數的SQL敘述句
}elseif(isset($_GET["keyword"])&&($_GET["keyword"]!="")){
	$query_RecProduct = "SELECT * FROM product WHERE productname LIKE ? OR description LIKE ? ORDER BY productid DESC";
	$stmt = $db_link->prepare($query_RecProduct);
	$keyword = "%".$_GET["keyword"]."%";
	$stmt->bind_param("ss", $keyword, $keyword);	
//若有價格區間關鍵字時未加限制顯示筆數的SQL敘述句
}elseif(isset($_GET["price1"]) && isset($_GET["price2"]) && ($_GET["price1"]<=$_GET["price2"])){
	$query_RecProduct = "SELECT * FROM product WHERE productprice BETWEEN ? AND ? ORDER BY productid DESC";
	$stmt = $db_link->prepare($query_RecProduct);
	$stmt->bind_param("ii", $_GET["price1"], $_GET["price1"]);
//預設狀況下未加限制顯示筆數的SQL敘述句
}else{
	$query_RecProduct = "SELECT * FROM product ORDER BY productid DESC";
	$stmt = $db_link->prepare($query_RecProduct);
}
$stmt->execute();
//以未加上限制顯示筆數的SQL敘述句查詢資料到 $all_RecProduct 中
$all_RecProduct = $stmt->get_result();
//計算總筆數
$total_records = $all_RecProduct->num_rows;
//計算總頁數=(總筆數/每頁筆數)後無條件進位。
$total_pages = ceil($total_records/$pageRow_records);
//繫結產品目錄資料
$query_RecCategory = "SELECT category.categoryid, category.categoryname, category.categorysort, count(product.productid) as productNum FROM category LEFT JOIN product ON category.categoryid = product.categoryid GROUP BY category.categoryid, category.categoryname, category.categorysort ORDER BY category.categorysort ASC";
$RecCategory = $db_link->query($query_RecCategory);
//計算資料總筆數
$query_RecTotal = "SELECT count(productid) as totalNum FROM product";
$RecTotal = $db_link->query($query_RecTotal);
$row_RecTotal = $RecTotal->fetch_assoc();
//返回 URL 參數
function keepURL(){
	$keepURL = "";
	if(isset($_GET["keyword"])) $keepURL.="&keyword=".urlencode($_GET["keyword"]);
	if(isset($_GET["price1"])) $keepURL.="&price1=".$_GET["price1"];
	if(isset($_GET["price2"])) $keepURL.="&price2=".$_GET["price2"];	
	if(isset($_GET["cid"])) $keepURL.="&cid=".$_GET["cid"];
	return $keepURL;
}
?>
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<title>網路購物系統</title>
<link href="style.css" rel="stylesheet" type="text/css">
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
                <li><a href="index.php">所有產品 <span class="categorycount">(<?php echo $row_RecTotal["totalNum"];?>)</span></a></li>
                <?php	while($row_RecCategory=$RecCategory->fetch_assoc()){ ?>
                <li><a href="index.php?cid=<?php echo $row_RecCategory["categoryid"];?>"><?php echo $row_RecCategory["categoryname"];?> <span class="categorycount">(<?php echo $row_RecCategory["productNum"];?>)</span></a></li>
                <?php }?>
              </ul>
            </div>
            <div class="boxbl"></div>
            <div class="boxbr"></div></td>
          <td><div class="subjectDiv"> <span class="heading"><img src="images/16-cube-green.png" width="16" height="16" align="absmiddle"></span> 產品列表</div>
            <div class="actionDiv"><a href="cart.php">我的購物車</a></div>
            <?php
            //加上限制顯示筆數的SQL敘述句，由本頁開始記錄筆數開始，每頁顯示預設筆數
            $query_limit_RecProduct = $query_RecProduct." LIMIT {$startRow_records}, {$pageRow_records}";
            //以加上限制顯示筆數的SQL敘述句查詢資料到 $RecProduct 中
            $stmt = $db_link->prepare($query_limit_RecProduct);
			//若有分類關鍵字時未加限制顯示筆數的SQL敘述句
			if(isset($_GET["cid"])&&($_GET["cid"]!="")){
				$stmt->bind_param("i", $_GET["cid"]);
			//若有搜尋關鍵字時未加限制顯示筆數的SQL敘述句
			}elseif(isset($_GET["keyword"])&&($_GET["keyword"]!="")){
				$keyword = "%".$_GET["keyword"]."%";
				$stmt->bind_param("ss", $keyword, $keyword);	
			//若有價格區間關鍵字時未加限制顯示筆數的SQL敘述句
			}elseif(isset($_GET["price1"]) && isset($_GET["price2"]) && ($_GET["price1"]<=$_GET["price2"])){
				$stmt->bind_param("ii", $_GET["price1"], $_GET["price2"]);
			}
            $stmt->execute();            
            $RecProduct = $stmt->get_result();
            while($row_RecProduct=$RecProduct->fetch_assoc()){ 
            ?>
            <div class="albumDiv">
              <div class="picDiv"><a href="product.php?id=<?php echo $row_RecProduct["productid"];?>">
                <?php if($row_RecProduct["productimages"]==""){?>
                <img src="images/nopic.png" alt="暫無圖片" width="120" height="120" border="0" />
                <?php }else{?>
                <img src="proimg/<?php echo $row_RecProduct["productimages"];?>" alt="<?php echo $row_RecProduct["productname"];?>" width="135" height="135" border="0" />
                <?php }?>
                </a></div>
              <div class="albuminfo"><a href="product.php?id=<?php echo $row_RecProduct["productid"];?>"><?php echo $row_RecProduct["productname"];?></a><br />
                <span class="smalltext">特價 </span><span class="redword"><?php echo $row_RecProduct["productprice"];?></span><span class="smalltext"> 元</span> </div>
            </div>
            <?php }?>
            <div class="navDiv">
              <?php if ($num_pages > 1) { // 若不是第一頁則顯示 ?>
              <a href="?page=1<?php echo keepURL();?>">|&lt;</a> <a href="?page=<?php echo $num_pages-1;?><?php echo keepURL();?>">&lt;&lt;</a>
              <?php }else{?>
              |&lt; &lt;&lt;
              <?php }?>
              <?php
  	  for($i=1;$i<=$total_pages;$i++){
  	  	  if($i==$num_pages){
  	  	  	  echo $i." ";
  	  	  }else{
  	  	      $urlstr = keepURL();
  	  	      echo "<a href=\"?page=$i$urlstr\">$i</a> ";
  	  	  }
  	  }
  	  ?>
              <?php if ($num_pages < $total_pages) { // 若不是最後一頁則顯示 ?>
              <a href="?page=<?php echo $num_pages+1;?><?php echo keepURL();?>">&gt;&gt;</a> <a href="?page=<?php echo $total_pages;?><?php echo keepURL();?>">&gt;|</a>
              <?php }else{?>
              &gt;&gt; &gt;|
              <?php }?>
            </div></td>
        </tr>
      </table></td>
  </tr>
  <tr>
    <td height="30" align="center" background="images/album_r2_c1.jpg" class="trademark">© 2016 eHappy Studio All Rights Reserved.</td>
  </tr>
</table>
</body>
</html>
<?php
$stmt->close();
$db_link->close();
?>