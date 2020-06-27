-- phpMyAdmin SQL Dump
-- version 4.5.1
-- http://www.phpmyadmin.net
--
-- 主機: 127.0.0.1
-- 產生時間： 2016-09-22 10:40:18
-- 伺服器版本: 10.1.13-MariaDB
-- PHP 版本： 7.0.8

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- 資料庫： `phpboard`
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
-- 資料表結構 `board`
--

CREATE TABLE `board` (
  `boardid` int(11) UNSIGNED NOT NULL,
  `boardname` varchar(50) COLLATE utf8_unicode_ci DEFAULT NULL,
  `boardsex` enum('男','女') COLLATE utf8_unicode_ci DEFAULT '男',
  `boardsubject` varchar(100) COLLATE utf8_unicode_ci DEFAULT NULL,
  `boardtime` datetime DEFAULT NULL,
  `boardmail` varchar(100) COLLATE utf8_unicode_ci DEFAULT NULL,
  `boardweb` varchar(100) COLLATE utf8_unicode_ci DEFAULT NULL,
  `boardcontent` text COLLATE utf8_unicode_ci
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_unicode_ci;

--
-- 資料表的匯出資料 `board`
--

INSERT INTO `board` (`boardid`, `boardname`, `boardsex`, `boardsubject`, `boardtime`, `boardmail`, `boardweb`, `boardcontent`) VALUES
(1, '茶米', '男', '這是第一則留言', '2016-10-16 11:25:30', 'david@e-happy.com.tw', 'http://www.e-happy.com.tw', '這是茶米的第一則留言！'),
(2, 'Lily', '女', '好棒喔，真不錯。', '2016-10-16 11:27:30', 'lily@e-happy.com.tw', '', '我也很喜歡這個留言版！'),
(3, '路人甲', '男', '我也來加油', '2016-10-16 11:36:51', '', 'http://www.e-happy.com.tw', '路過，我也來加油喔！\r\n灌水灌水灌水灌水灌水灌水灌水灌水灌水灌水灌水灌水灌水灌水灌水灌水灌水灌水灌水灌水。'),
(4, '路人乙', '男', '我也要學！', '2016-10-16 11:38:23', '', '', '請問這個程式很難嗎？\r\n我也好想學喔！'),
(5, '學員A', '女', '好漂亮的留言版', '2016-10-16 11:39:47', '', '', '這個留言版好漂亮喔！'),
(6, '學員B', '女', '這是我第一次留言', '2016-10-16 11:40:25', '', '', '這是我第一次留言，\r\n希望大家多多指教。'),
(7, '茶米', '男', '感謝大家的測試！', '2016-10-16 11:47:12', 'david@e-happy.com.tw', 'http://www.e-happy.com.tw', '感謝大家的測試！\r\n也希望大家多多棒場。');

--
-- 已匯出資料表的索引
--

--
-- 資料表索引 `admin`
--
ALTER TABLE `admin`
  ADD PRIMARY KEY (`username`);

--
-- 資料表索引 `board`
--
ALTER TABLE `board`
  ADD PRIMARY KEY (`boardid`);

--
-- 在匯出的資料表使用 AUTO_INCREMENT
--

--
-- 使用資料表 AUTO_INCREMENT `board`
--
ALTER TABLE `board`
  MODIFY `boardid` int(11) UNSIGNED NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=8;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
