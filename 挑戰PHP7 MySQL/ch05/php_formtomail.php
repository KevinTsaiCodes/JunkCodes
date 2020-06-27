1	<?php
2	if(isset($_POST["sendmail"]) && ($_POST["sendmail"]=="true")){
3		$mailFrom="=?UTF-8?B?" . base64_encode($_POST["fromname"]) . "?= <".$_POST["frommail"].">";
4		$mailto="david@e-happy.com.tw";
5		$mailSubject="=?UTF-8?B?" . base64_encode($_POST["mailsubject"]). "?=";
6		$mailHeader="From:".$mailFrom."\r\n";
7		$mailHeader.="Content-type:text/html;charset=UTF-8";
8		mail($mailto,$mailSubject,$_POST["mailcontent"],$mailHeader);
9	}
10	?>
11	<!DOCTYPE html>
12	<html>
13	<head>
14	  <meta charset="UTF-8">
15	  <title>聯絡表單</title>
16	</head>
17	<body>
18	<?php if(isset($_POST["sendmail"]) && ($_POST["sendmail"]=="true")){?>
19	寄送成功，<a href="php_formtomail.php">再寄一次</a>。
20	<?php }else{?>
21	<form name="form1" method="post" action="">
22	  <table border="0" cellpadding="4" cellspacing="0">
23	    <tr>
24	      <td>寄件人姓名</td>
25	      <td><input type="text" name="fromname" id="fromname"></td>
26	    </tr>
27	    <tr>
28	      <td>寄件人郵件</td>
29	      <td><input type="text" name="frommail" id="frommail"></td>
30	    </tr>
31	    
32	    <tr>
33	      <td>主旨</td>
34	      <td><input type="text" name="mailsubject" id="mailsubject"></td>
35	    </tr>
36	    <tr>
37	      <td colspan="2"><textarea name="mailcontent" id="mailcontent" cols="45" rows="5"></textarea></td>
38	    </tr>
39	    <tr>
40	      <td colspan="2"><input type="submit" name="button" id="button" value="送出">
41	      <input name="sendmail" type="hidden" id="sendmail" value="true"></td>
42	    </tr>
43	  </table>
44	</form>
45	<?php }?>
46	</body>
47	</html>	