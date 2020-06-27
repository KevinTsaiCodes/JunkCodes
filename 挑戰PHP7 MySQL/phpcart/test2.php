<?php
require_once("connMysql.php");
//預設每頁筆數
$pageRow_records = 3;
//預設頁數
$num_pages = 1;
//若已經有翻頁，將頁數更新
if (isset($_GET['page'])) {
  $num_pages = $_GET['page'];
}
//本頁開始記錄筆數 = (頁數-1)*每頁記錄筆數
$startRow_records = ($num_pages -1) * $pageRow_records;
$cid=2;
$id=1;
$query_RecProduct = "SELECT * FROM product WHERE categoryid=? ORDER BY productid DESC";
//加上限制顯示筆數的SQL敘述句，由本頁開始記錄筆數開始，每頁顯示預設筆數
$query_limit_RecProduct = $query_RecProduct." LIMIT {$startRow_records}, {$pageRow_records}";
$stmt = $db_link->prepare($query_RecProduct);

$stmt->bind_param("i", $cid);
$stmt->execute();
$result = $stmt->get_result();
echo $result->num_rows."<br>";


$stmt = $db_link->prepare($query_limit_RecProduct);
$stmt->bind_param("i", $id);
$stmt->execute();
$result = $stmt->get_result();

while ($row = $result->fetch_assoc()) {
	printf("%s %s %s %s %s<br>", $row["productid"], $row["categoryid"], $row["productname"], $row["productprice"], $row["productimages"]);
}

$stmt->close();
?>