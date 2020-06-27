<?php
require_once("connMysql.php");
session_start();
//檢查是否經過登入
if(!isset($_SESSION["loginMember"]) || ($_SESSION["loginMember"]=="")){
	header("Location: index.php");
}
//執行登出動作
if(isset($_GET["logout"]) && ($_GET["logout"]=="true")){
	unset($_SESSION["loginMember"]);
	header("Location: index.php");
}
//執行刪除動作
if(isset($_POST["action"])&&($_POST["action"]=="delete")){	
	$sql_query = "DELETE FROM board WHERE boardid=?";
	$stmt=$db_link->prepare($sql_query);
	$stmt->bind_param("i",$_POST["boardid"]);
	$stmt->execute();
	$stmt->close();
	//重新導向回到主畫面
	header("Location: admin.php");
}
$query_RecBoard = "SELECT boardid, boardname, boardsex, boardsubject, boardmail, boardweb, boardcontent FROM board WHERE boardid=?";
$stmt=$db_link->prepare($query_RecBoard);
$stmt->bind_param("i", $_GET["id"]);
$stmt->execute();
$stmt->bind_result($boardid, $boardname, $boardsex, $boardsubject, $boardmail, $boardweb, $boardcontent);
$stmt->fetch();
?>
<html>
<head>
<title>訪客留言版管理系統</title>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
<link href="style.css" rel="stylesheet" type="text/css">
</head>
<body bgcolor="#ffffff">
<table width="700" border="0" align="center" cellpadding="0" cellspacing="0">
  <tr>
    <td><table align="left" border="0" cellpadding="0" cellspacing="0" width="700">
        <tr>
          <td background="images/admin_topbg.jpg"><img name="admin_r1_c1" src="images/admin_r1_c1.jpg" width="465" height="36" border="0" alt=""></td>
          <td width="15"><img name="admin_r1_c8" src="images/admin_r1_c8.jpg" width="15" height="36" border="0" alt=""></td>
        </tr>
    </table></td>
  </tr>
  <tr>
    <td><img name="admin_r2_c1" src="images/admin_r2_c1.jpg" width="700" height="28" border="0" alt=""></td>
  </tr>
  <tr>
    <td background="images/admin_r3_c1.jpg"><div id="mainRegion">
        <form name="form1" method="post" action="">
          <table width="90%" border="0" align="center" cellpadding="4" cellspacing="0">
            <tr valign="top">
              <td class="heading">刪除訪客留言版資料</td>
            </tr>
            <tr valign="top">
              <td>
                <p><strong>標題</strong>：<?php echo $boardsubject;?> <strong>姓名</strong>：<?php echo $boardname;?> <strong>性別</strong>：<?php echo $boardsex;?></p>
                <p><strong>郵件</strong>：<?php echo $boardmail;?> <strong>網站</strong>：<?php echo $boardweb;?></p>
                <p><?php echo nl2br($boardcontent);?></p>
              </td>
            </tr>
            <tr valign="top">
              <td align="center"><p>
                  <input name="boardid" type="hidden" id="boardid" value="<?php echo $boardid;?>">
                  <input name="action" type="hidden" id="action" value="delete">
                  <input type="submit" name="button" id="button" value="確定刪除資料">
                  <input type="button" name="button3" id="button3" value="回上一頁" onClick="window.history.back();">
                </p></td>
            </tr>
          </table>
        </form>
      </div></td>
  </tr>
  <tr>
    <td><table align="left" border="0" cellpadding="0" cellspacing="0" width="700">
        <tr>
          <td width="15"><img name="admin_r4_c1" src="images/admin_r4_c1.jpg" width="15" height="31" border="0" alt=""></td>
          <td background="images/admin_botbg.jpg"><a href="?logout=true"><img name="admin_r4_c2" src="images/logout.jpg" width="77" height="31" border="0" alt="登出管理"></a></td>
          <td align="right" valign="top" background="images/admin_botbg.jpg" class="trademark">© 2016 eHappy Studio All Rights Reserved. </td>
          <td width="15"><img name="admin_r4_c8" src="images/admin_r4_c8.jpg" width="15" height="31" border="0" alt=""></td>
        </tr>
      </table></td>
  </tr>
</table>
</body>
</html>
<?php
	$stmt->close();
	$db_link->close();
?>