<?php 
if(isset($_GET["question"])&&($_GET["question"]!="")){
	if(isset($_GET["answer1"])&&($_GET["answer1"]!="")){
		setcookie("answer1",$_GET["answer1"],time()+3600);
		header("Location: lesson10.php?step=2");
	}
	if(isset($_GET["answer2"])&&($_GET["answer2"]!="")){
		setcookie("answer2",$_GET["answer2"],time()+3600);
		header("Location: lesson10.php?step=3");
	}
	if(isset($_GET["answer3"])&&($_GET["answer3"]!="")){
		setcookie("answer3",$_GET["answer3"],time()+3600);
		header("Location: lesson10.php?step=4");
	}
	if(isset($_GET["answer4"])&&($_GET["answer4"]!="")){
		setcookie("answer4",$_GET["answer4"],time()+3600);
		header("Location: lesson10.php?result=true");
	}			
}
if(isset($_GET["restart"])&&($_GET["restart"]!="")){
	setcookie("answer1","",time()-3600);
	setcookie("answer2","",time()-3600);
	setcookie("answer3","",time()-3600);
	setcookie("answer4","",time()-3600);
}
?>
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<title>線上心理測驗</title>
</head>
<body>
<p>線上心理測驗：</p>
<?php if(($_SERVER['QUERY_STRING']=="")|| isset($_GET["restart"])){?>
<p><a href="lesson10.php?step=1">準備好了嗎？請按此開始測驗！</a></p>
<?php }?>
<?php if(isset($_GET["step"])&&($_GET["step"]==1)){?>
<p>假如你是一支兔子：</p>
<form id="form1" name="form1" method="get" action="">
  <p>1.有一天你經過一座橋，橋下風景很漂亮，你會：</p>
  <p><input name="answer1" type="radio" id="radio" value="1" checked="checked" />
    用慢慢走的過去<br />
    <input type="radio" name="answer1" id="radio2" value="2" />
    開開心心蹦蹦跳跳過去<br />
    <input type="radio" name="answer1" id="radio3" value="3" />
    急急忙忙的跑過去</p>
  <p>
    <input type="submit" id="button" value="下一步" />
    <input name="question" type="hidden" id="question" value="true">
  </p>
</form>
<?php }else if(isset($_GET["step"])&&($_GET["step"]=="2")){?>
<form id="form2" name="form2" method="get" action="">
  <p>2.中途遇見一隻兔子，長的漂漂亮亮，躺在路邊一動也不動的，你心裡想：</p>
  <p><input name="answer2" type="radio" id="radio4" value="1" checked="checked" />
    他暈倒了<br />
    <input type="radio" name="answer2" id="radio5" value="2" />
    他在休息<br />
    <input type="radio" name="answer2" id="radio6" value="3" />
    他死了</p>
  <p>
    <input type="submit" id="button2" value="下一步" />
    <input name="question" type="hidden" id="question" value="true">
  </p>
</form>
<?php }else if(isset($_GET["step"])&&($_GET["step"]=="3")){?>
<form id="form3" name="form3" method="get" action="">
  <p>3.過橋一半的時候，你回家的唯一鑰匙掉到橋下，你會：  </p>
  <p>
    <input name="answer3" type="radio" id="radio7" value="1" checked="checked" />
    你會奮不顧身的下去撿<br />
    <input type="radio" name="answer3" id="radio8" value="2" />
    在橋上觀望再下去揀<br />
    <input type="radio" name="answer3" id="radio9" value="3" />
    揀都不揀就回家了</p>
  <p>
    <input type="submit" id="button3" value="下一步" />
    <input name="question" type="hidden" id="question" value="true">
  </p>
</form>
<?php }else if(isset($_GET["step"])&&($_GET["step"]=="4")){?>
<form id="form4" name="form4" method="get" action="">
  <p>4.過橋一半的時候，你回家的唯一鑰匙掉到橋下，你會： </p>
  <p>
    <input name="answer4" type="radio" id="radio10" value="1" checked="checked" />
    回到家你沒有鑰匙你會怎麼回家<br />
    <input type="radio" name="answer4" id="radio11" value="2" />
    找可能的入口翻進去<br />
    <input type="radio" name="answer4" id="radio12" value="3" />
    等家人回來開門</p>
  <p>
    <input type="submit" id="button4" value="看測驗結果" />
    <input name="question" type="hidden" id="question" value="true">
  </p>
</form>
<?php }?>
<p>
  <?php if(isset($_GET["result"])&&($_GET["result"]=="true")){?>
</p>
<p>分析結果：</p>
<p>【人生觀態度】</p>
<?php 
if(isset($_COOKIE["answer1"])&&($_COOKIE["answer1"]!="")){
	switch ($_COOKIE["answer1"]){
	case "1":
		echo "<p>你對自己未來沒有把握！總是邊走邊看走一步算一步的那種！</p>";
		break;
	case "2":	
		echo "<p>你對自己的未來充滿自信，相信你這一生不會白活才對！</p>";
		break;
	case "3":		
		echo "<p>你是為了別人而活！忙忙碌碌過一生，有點可憐！</p>";
		break;
	}	
}
?>
<p>【愛情觀】</p>
<?php 
if(isset($_COOKIE["answer2"])&&($_COOKIE["answer2"]!="")){
	switch ($_COOKIE["answer2"]){
	case "1":
		echo "<p>你可能對你的愛情不能專心，常常半路就會放棄跑去睡覺！</p>";
		break;
	case "2":	
		echo "<p>你遇到愛情會無法自拔，會墜入愛情的迷網裡！頭腦暈暈的。</p>";
		break;
	case "3":		
		echo "<p>你對愛沒啥渴望，心中缺乏愛。</p>";
		break;
	}	
}
?>
<p>【對金錢的看法】</p>
<?php 
if(isset($_COOKIE["answer3"])&&($_COOKIE["answer3"]!="")){
	switch ($_COOKIE["answer3"]){
	case "1":
		echo "<p>你貪財如命，會為了錢誤了你的一生。</p>";
		break;
	case "2":	
		echo "<p>你對金錢管理會謹慎一點，常會想買一個東西而想好久才買，買了不喜歡的會難過好久！</p>";
		break;
	case "3":		
		echo "<p>你對金錢比較不在乎，視金錢如糞土的人用金錢收買不了他的心的！</p>";
		break;
	}	
}
?>
<p>【對家的感覺！包括對家人的態度】</p>
<?php 
if(isset($_COOKIE["answer4"])&&($_COOKIE["answer4"]!="")){
	switch ($_COOKIE["answer4"]){
	case "1":
		echo "<p>你對家沒甚麼依賴度 如果發生事情你會找朋友而不會想到家人。</p>";
		break;
	case "2":	
		echo "<p>你對家人的態度有時會很不耐煩，但事後又會很後悔，但你還是會想家的。</p>";
		break;
	case "3":		
		echo "<p>你的家庭一定很美滿，你跟家人處的很好，彼此會互相照顧，是個很顧家的人！</p>";
		break;
	}	
}
?>
<p><a href="lesson10.php?restart=true">重來一次!</a></p>
<?php }?>
</body>
</html>