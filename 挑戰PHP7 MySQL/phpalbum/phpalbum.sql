-- phpMyAdmin SQL Dump
-- version 4.5.1
-- http://www.phpmyadmin.net
--
-- 主機: 127.0.0.1
-- 產生時間： 2016-09-22 10:40:47
-- 伺服器版本: 10.1.13-MariaDB
-- PHP 版本： 7.0.8

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- 資料庫： `phpalbum`
--

-- --------------------------------------------------------

--
-- 資料表結構 `admin`
--

CREATE TABLE `admin` (
  `username` varchar(50) COLLATE utf8_unicode_ci NOT NULL DEFAULT '',
  `passwd` varchar(50) COLLATE utf8_unicode_ci NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;

--
-- 資料表的匯出資料 `admin`
--

INSERT INTO `admin` (`username`, `passwd`) VALUES
('admin', 'admin');

-- --------------------------------------------------------

--
-- 資料表結構 `album`
--

CREATE TABLE `album` (
  `album_id` int(11) UNSIGNED NOT NULL,
  `album_date` datetime DEFAULT NULL,
  `album_location` varchar(255) COLLATE utf8_unicode_ci DEFAULT NULL,
  `album_title` varchar(255) COLLATE utf8_unicode_ci DEFAULT NULL,
  `album_desc` text COLLATE utf8_unicode_ci
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;

--
-- 資料表的匯出資料 `album`
--

INSERT INTO `album` (`album_id`, `album_date`, `album_location`, `album_title`, `album_desc`) VALUES
(1, '2016-10-20 12:11:53', '清水', '清水路邊老攤冰店', '清水路邊老攤冰店，很古早味的冰。'),
(2, '2016-10-20 12:13:08', '樹太老定食', '一家三口聚餐', '樹太老定食，不錯吃。'),
(3, '2016-10-20 12:14:37', '大翔書局', '十月份慶生', '幫朋友小朋友十月份慶生'),
(4, '2016-10-20 12:15:16', '三育書院', '三育書院', '三育書院很漂亮，假日可以進去逛逛。'),
(5, '2016-10-20 12:16:18', '文淵閣工作室', '忙碌的工作室', '大家忙著做卡片，很忙啊！'),
(6, '2016-10-20 12:17:22', '清水牛肉麵', '兒子吃牛肉麵', '清水牛肉麵，很好吃，看兒子的樣子就知道了。'),
(7, '2016-10-20 12:18:31', '埔里往武嶺的路上', '武嶺單車之旅', '武嶺，我來了。'),
(8, '2016-10-20 12:22:39', '高美溼地', '高美溼地', '高美溼地，怎麼拍都好看。'),
(9, '2016-10-20 12:24:31', '各處', '可愛的兒子', '嗯，真是可愛，誰生的嘛～～～');

-- --------------------------------------------------------

--
-- 資料表結構 `albumphoto`
--

CREATE TABLE `albumphoto` (
  `ap_id` int(11) UNSIGNED NOT NULL,
  `album_id` int(11) UNSIGNED DEFAULT NULL,
  `ap_subject` varchar(255) COLLATE utf8_unicode_ci DEFAULT NULL,
  `ap_date` datetime DEFAULT NULL,
  `ap_picurl` varchar(100) COLLATE utf8_unicode_ci DEFAULT NULL,
  `ap_hits` int(11) UNSIGNED NOT NULL DEFAULT '0'
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;

--
-- 資料表的匯出資料 `albumphoto`
--

INSERT INTO `albumphoto` (`ap_id`, `album_id`, `ap_subject`, `ap_date`, `ap_picurl`, `ap_hits`) VALUES
(1, 1, '清水路邊老攤冰店', '2016-10-20 12:13:03', 'IMAGE_042.jpg', 0),
(2, 1, '清水路邊老攤冰店', '2016-10-20 12:13:03', 'IMAGE_043.jpg', 0),
(3, 1, '清水路邊老攤冰店', '2016-10-20 12:13:03', 'IMAGE_044.jpg', 0),
(4, 1, '清水路邊老攤冰店', '2016-10-20 12:13:03', 'IMAGE_045.jpg', 0),
(5, 2, '樹太老定食', '2016-10-20 12:13:48', 'IMAGE_052.jpg', 0),
(6, 2, '樹太老定食', '2016-10-20 12:13:48', 'IMAGE_053.jpg', 0),
(7, 2, '樹太老定食', '2016-10-20 12:13:48', 'IMAGE_054.jpg', 0),
(8, 2, '樹太老定食', '2016-10-20 12:13:49', 'IMAGE_058.jpg', 0),
(9, 2, '樹太老定食', '2016-10-20 12:13:49', 'IMAGE_059.jpg', 0),
(10, 2, '樹太老定食', '2016-10-20 12:14:24', 'IMAGE_061.jpg', 0),
(11, 2, '樹太老定食', '2016-10-20 12:14:24', 'IMAGE_062.jpg', 0),
(12, 3, '十月份慶生', '2016-10-20 12:15:12', '201609202004_076.jpg', 0),
(13, 3, '十月份慶生', '2016-10-20 12:15:12', '201609202004_077.jpg', 0),
(14, 4, '三育書院', '2016-10-20 12:16:06', 'IMAGE_00073.jpg', 0),
(15, 4, '三育書院', '2016-10-20 12:16:06', 'IMAGE_00075.jpg', 0),
(16, 4, '三育書院', '2016-10-20 12:16:06', 'IMAGE_00076.jpg', 0),
(17, 4, '三育書院', '2016-10-20 12:16:06', 'IMAGE_00078.jpg', 0),
(18, 4, '三育書院', '2016-10-20 12:16:06', 'IMAGE_00079.jpg', 0),
(19, 5, '忙碌的工作室', '2016-10-20 12:17:20', 'IMAGE_011.jpg', 0),
(20, 5, '忙碌的工作室', '2016-10-20 12:17:20', 'IMAGE_012.jpg', 0),
(21, 5, '忙碌的工作室', '2016-10-20 12:17:20', 'IMAGE_013.jpg', 0),
(22, 5, '忙碌的工作室', '2016-10-20 12:17:20', 'IMAGE_015.jpg', 0),
(23, 6, '清水牛肉麵', '2016-10-20 12:18:28', 'IMAGE_034.jpg', 0),
(24, 6, '清水牛肉麵', '2016-10-20 12:18:28', 'IMAGE_035.jpg', 0),
(25, 6, '清水牛肉麵', '2016-10-20 12:18:28', 'IMAGE_041.jpg', 0),
(26, 6, '清水牛肉麵', '2016-10-20 12:18:28', 'IMAGE_048.jpg', 0),
(27, 7, '武嶺單車之旅', '2016-10-20 12:19:50', 'DSC09616.JPG', 0),
(28, 7, '武嶺單車之旅', '2016-10-20 12:19:50', 'DSC09627.JPG', 0),
(29, 7, '武嶺單車之旅', '2016-10-20 12:19:50', 'DSC09631.JPG', 0),
(30, 7, '武嶺單車之旅', '2016-10-20 12:19:50', 'DSC09678.JPG', 0),
(31, 7, '武嶺單車之旅', '2016-10-20 12:19:50', 'DSC09685.JPG', 0),
(32, 7, '武嶺單車之旅', '2016-10-20 12:20:18', 'DSC09689.JPG', 0),
(33, 7, '武嶺單車之旅', '2016-10-20 12:20:18', 'DSC09692.JPG', 0),
(34, 7, '武嶺單車之旅', '2016-10-20 12:20:18', 'DSC09695.JPG', 0),
(35, 8, '高美溼地', '2016-10-20 12:23:11', 'IMAGE_049.jpg', 0),
(36, 8, '高美溼地', '2016-10-20 12:23:11', 'IMAGE_050.jpg', 0),
(37, 8, '高美溼地', '2016-10-20 12:23:11', 'IMAGE_051.jpg', 0),
(38, 9, '可愛的兒子', '2016-10-20 12:25:25', '201609201134_072.jpg', 0),
(39, 9, '可愛的兒子', '2016-10-20 12:25:25', 'DSCN3442.JPG', 0),
(40, 9, '可愛的兒子', '2016-10-20 12:25:25', 'DSCN3449.JPG', 0),
(41, 9, '可愛的兒子', '2016-10-20 12:25:25', 'DSCN3562.JPG', 0),
(42, 9, '可愛的兒子', '2016-10-20 12:25:25', 'DSCN3693.JPG', 0),
(43, 9, '可愛的兒子', '2016-10-20 12:25:44', 'IMAGE_00038.jpg', 0),
(44, 9, '可愛的兒子', '2016-10-20 12:25:44', 'IMAGE_00040.jpg', 0),
(45, 9, '可愛的兒子', '2016-10-20 12:25:44', 'IMAGE_00135.jpg', 0);

--
-- 已匯出資料表的索引
--

--
-- 資料表索引 `admin`
--
ALTER TABLE `admin`
  ADD PRIMARY KEY (`username`);

--
-- 資料表索引 `album`
--
ALTER TABLE `album`
  ADD PRIMARY KEY (`album_id`);

--
-- 資料表索引 `albumphoto`
--
ALTER TABLE `albumphoto`
  ADD PRIMARY KEY (`ap_id`);

--
-- 在匯出的資料表使用 AUTO_INCREMENT
--

--
-- 使用資料表 AUTO_INCREMENT `album`
--
ALTER TABLE `album`
  MODIFY `album_id` int(11) UNSIGNED NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=10;
--
-- 使用資料表 AUTO_INCREMENT `albumphoto`
--
ALTER TABLE `albumphoto`
  MODIFY `ap_id` int(11) UNSIGNED NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=46;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
