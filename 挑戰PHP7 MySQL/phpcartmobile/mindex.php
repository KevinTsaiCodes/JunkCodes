<?php
require_once("connMysql.php");
//繫結產品目錄資料
$query_RecCategory = "SELECT category.categoryid, category.categoryname, category.categorysort, count(product.productid) as productNum FROM category LEFT JOIN product ON category.categoryid = product.categoryid GROUP BY category.categoryid, category.categoryname, category.categorysort ORDER BY category.categorysort ASC";
$RecCategory = mysqli_query($db_link, $query_RecCategory);
?>
<!DOCTYPE html> 
<html>
<head>
<meta charset="utf-8">
<title>PHP行動購物網</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link href="http://code.jquery.com/mobile/1.4.5/jquery.mobile-1.4.5.min.css" rel="stylesheet" type="text/css"/>
<script src="http://code.jquery.com/jquery-1.11.1.min.js" type="text/javascript"></script>
<script src="http://code.jquery.com/mobile/1.4.5/jquery.mobile-1.4.5.min.js" type="text/javascript"></script>
</head> 
<body> 

<div data-role="page" id="page" data-add-back-btn="true">
	<div data-position="fixed" data-role="header" data-position="fixed">
		<h1>行動購物網</h1>
	</div>
	<div data-role="content">	
		<ul data-role="listview" data-inset="true" data-filter="true">
			<li data-role="list-divider">品牌</li>
            <?php	while($row_RecCategory=mysqli_fetch_assoc($RecCategory)){ ?>
            <li><a href="mcategory.php?cid=<?php echo $row_RecCategory["categoryid"];?>"><?php echo $row_RecCategory["categoryname"];?><span class="ui-li-count"><?php echo $row_RecCategory["productNum"];?></span></a></li>
            <?php }?>
		</ul>		
	</div>
	<div data-role="footer" data-position="fixed">
		<div data-role="navbar">
            <ul>
                <li><a href="mindex.php" data-icon="grid" class="ui-btn-active">逛逛商店</a></li>
                <li><a href="mcart.php" data-icon="star" data-ajax="false">檢視購物車</a></li>
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
</div>
</body>
</html>
<?php
$db_link->close();
?>