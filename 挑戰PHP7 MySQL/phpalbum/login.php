<?php
session_start();
//如果沒有登入Session值或是Session值為空則執行登入動作
if(!isset($_SESSION["loginMember"]) || ($_SESSION["loginMember"]=="")){
	if(isset($_POST["username"]) && isset($_POST["passwd"])){
		require_once("connMysql.php");		
		//選取儲存帳號密碼的資料表
		$sql_query = "SELECT * FROM admin";
		$result = $db_link->query($sql_query);		
		//取出帳號密碼的值
		$row_result=$result->fetch_assoc();
		$username = $row_result["username"];
		$passwd = $row_result["passwd"];
		$db_link->close();
		//比對帳號密碼，若登入成功則進往管理界面，否則就退回主畫面。
		if(($_POST["username"]==$username) && ($_POST["passwd"]==$passwd)){
			$_SESSION["loginMember"]=$username;
			header("Location: admin.php");
		}else{
			header("Location: index.php");
		}		
	}
}else{
	//若已經有登入Session值則前往管理界面
	header("Location: admin.php");
}
?>
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<title>網路相簿</title>
<link href="style.css" rel="stylesheet" type="text/css" />
</head>
<body bgcolor="#cccccc">
<table width="778" border="0" align="center" cellpadding="0" cellspacing="0">
  <tr>
   <td height="124" valign="top" background="images/album_r1_c1.jpg"><div class="titleDiv"> [生活、旅行的記錄]<br />
   </div>
    <div class="menulink"><a href="index.php">[相簿首頁]</a> <a href="login.php">[相簿管理]</a></div></td>
  </tr>
  <tr>
   <td background="images/album_r2_c1.jpg"><div id="mainRegion">
     <table width="90%" border="0" align="center" cellpadding="4" cellspacing="0">
       <tr>
         <td><div class="subjectDiv"> 網路相簿管理登入 </div>             
           <div class="albumdesc">
             <form id="form1" name="form1" method="post" action="">
               <table border="0" align="center" cellpadding="4" cellspacing="0">
                 <tr>
                   <td><p>管理者帳號</p></td>
                   <td><p>
                     <input type="text" name="username" id="username" />
                   </p></td>
                 </tr>
                 <tr>
                   <td><p>管理者密碼</p></td>
                   <td><p>
                     <input type="password" name="passwd" id="passwd" />
                   </p></td>
                 </tr>
                 <tr>
                   <td colspan="2" align="center"><p>
                     <input type="submit" name="button" id="button" value="登入管理" />
                     <input type="button" name="button2" id="button2" value="回上一頁" onClick="window.history.back();" />
                   </p></td>
                   </tr>
               </table>
                </form>
             </div>
           </td>
       </tr>
     </table>
   </div></td>
  </tr>
  <tr>
   <td align="center" background="images/album_r2_c1.jpg" class="trademark">© 2016 eHappy Studio All Rights Reserved.</td>
  </tr>
  <tr>
   <td><img name="album_r4_c1" src="images/album_r4_c1.jpg" width="778" height="24" border="0" id="album_r4_c1" alt="" /></td>
  </tr>
</table>
</body>
</html>