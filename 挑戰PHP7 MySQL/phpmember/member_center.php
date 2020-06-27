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
	unset($_SESSION["memberLevel"]);
	header("Location: index.php");
}
//繫結登入會員資料
$query_RecMember = "SELECT * FROM memberdata WHERE m_username = '{$_SESSION["loginMember"]}'";
$RecMember = $db_link->query($query_RecMember);	
$row_RecMember=$RecMember->fetch_assoc();
?>
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<title>網站會員系統</title>
<link href="style.css" rel="stylesheet" type="text/css">
</head>

<body>
<table width="780" border="0" align="center" cellpadding="4" cellspacing="0">
  <tr>
    <td class="tdbline"><img src="images/mlogo.png" alt="會員系統" width="164" height="67"></td>
  </tr>
  <tr>
    <td class="tdbline"><table width="100%" border="0" cellspacing="0" cellpadding="10">
      <tr valign="top">
        <td class="tdrline"><p class="title">歡迎光臨網站會員系統</p>
          <p>感謝各位來到會員系統， 所有的會員功能都必須經由登入後才能使用，請您在右方視窗中執行登入動作。</p>
          <p class="heading"> 本會員系統擁有以下的功能：</p>
          <ol>
            <li>免費加入會員 。</li>
            <li>每個會員可修改本身資料。</li>
            <li>若是遺忘密碼，會員可由系統發出電子信函通知。</li>
            <li>管理者可以修改、刪除會員的資料。</li>
          </ol>
          <p class="heading">請各位會員遵守以下規則： </p>
          <ol>
            <li> 遵守政府的各項有關法律法規。</li>
            <li> 不得在發佈任何色情非法， 以及危害國家安全的言論。</li>
            <li>嚴禁連結有關政治， 色情， 宗教， 迷信等違法訊息。</li>
            <li> 承擔一切因您的行為而直接或間接導致的民事或刑事法律責任。</li>
            <li> 互相尊重， 遵守互聯網絡道德；嚴禁互相惡意攻擊， 漫罵。</li>
            <li> 管理員擁有一切管理權力。</li>
          </ol></td>
        <td width="200">
        <div class="boxtl"></div><div class="boxtr"></div>
<div class="regbox">
          <p class="heading"><strong>會員系統</strong></p>
          
            <p><strong><?php echo $row_RecMember["m_name"];?></strong> 您好。</p>
            <p>您總共登入了 <?php echo $row_RecMember["m_login"];?> 次。<br>
            本次登入的時間為：<br>
            <?php echo $row_RecMember["m_logintime"];?></p>
            <p align="center"><a href="member_update.php">修改資料</a> | <a href="?logout=true">登出系統</a></p>
</div>
        <div class="boxbl"></div><div class="boxbr"></div></td>
      </tr>
    </table></td>
  </tr>
  <tr>
    <td align="center" background="images/album_r2_c1.jpg" class="trademark">© 2016 eHappy Studio All Rights Reserved.</td>
  </tr>
</table>
</body>
</html>
<?php
	$db_link->close();
?>