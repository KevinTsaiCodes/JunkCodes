<?php
require_once("connMysql.php");
$query_RecProduct = "SELECT * FROM product WHERE categoryid=? ORDER BY productid DESC";
$stmt = $db_link->prepare($query_RecProduct);
$cid=1;
$stmt->bind_param("i", $cid);
$stmt->execute();

/*
預存語法使用 get_result() 的方式取得資料後生成資料物件
再由物件的num_rows屬性可以取得資料總數
接著可由fetch_assoc()方法取得以欄位為名的資料內容
*/
$result = $stmt->get_result();
echo $result->num_rows."<br>";

while ($row = $result->fetch_assoc()) {
	printf("%s %s %s %s %s<br>", $row["productid"], $row["categoryid"], $row["productname"], $row["productprice"], $row["productimages"]);
}

echo(memory_get_usage());
echo "<br>";
//$stmt->close();
echo(memory_get_usage());
echo "<br>";

//$query_RecProduct = "SELECT * FROM product WHERE categoryid=? ORDER BY productid DESC";
//$stmt = $db_link->prepare($query_RecProduct);
$cid=2;
$stmt->bind_param("i", $cid);
$stmt->execute();
/*
預存語法使用 store_result() 將stmt資料內容暫存在記憶體中
再由stmt物件的num_rows屬性可以取得資料總數
接著可由bind_result()、fetch()方法取得以綁定變數的資料內容
*/
$stmt->store_result();
echo $stmt->num_rows."<br>";
$stmt->bind_result($pid, $cid, $pname, $pprice, $pimg, $pdesc);
while ($stmt->fetch()){
	printf("%s %s %s %s %s<br>", $pid, $cid, $pname, $pprice, $pimg);	
}

echo(memory_get_usage());
echo "<br>";
$stmt->free_result();
echo(memory_get_usage());
echo "<br>";
$stmt->close();
echo(memory_get_usage());
echo "<br>";
$db_link->close();
echo(memory_get_usage());
echo "<br>";
?>